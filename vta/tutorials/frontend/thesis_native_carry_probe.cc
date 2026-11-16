// Exercise the real native LoadStage/RunStage and GraphExecutor storage semantics.
#define main ThesisNativeProductionMain
#include "vta_stage_pipeline_runner.cc"
#undef main

int main(int argc, char** argv) {
  try {
    if (argc != 8) throw std::runtime_error("expected graph lib params inputs output-directory mode input-names");
    StageArgs arg;
    arg.name = "carry_probe"; arg.device = "cpu";
    arg.graph = argv[1]; arg.lib = argv[2]; arg.params = argv[3];
    arg.input_names = SplitCSV(argv[7]); arg.cpu_input_zero_copy = true;
    StageExecutor stage = LoadStage(arg.name, arg.device, arg);
    const bool naive = std::string(argv[6]) == "naive";
    const auto paths = SplitCSV(argv[4]);
    if (paths.size() != arg.input_names.size()) throw std::runtime_error("input arity mismatch");
    std::vector<NDArray> inputs;
    for (size_t i = 0; i < paths.size(); ++i) {
      NDArray original = stage.get_input(arg.input_names[i]).operator NDArray();
      const auto bytes = ReadFile(paths[i], true);
      inputs.push_back(MakeCPUInputLike(original.operator->(), bytes));
      std::string poison = bytes;
      for (char& byte : poison) byte = static_cast<char>(~static_cast<unsigned char>(byte));
      NDArray poisoned = MakeCPUInputLike(original.operator->(), poison);
      original.CopyFrom(poisoned);
      if (std::memcmp(TensorDataBytes(original.operator->()), poison.data(), poison.size()) != 0)
        throw std::runtime_error("original pool poison did not persist");
    }
    int carry_count = 0;
    for (int input_index : stage.output_live_input_indices) if (input_index >= 0) ++carry_count;
    if (carry_count != 1) throw std::runtime_error("probe needs exactly one actual live-input head");
    if (naive) stage.output_live_input_indices.assign(stage.NumOutputs(), -1);
    StageTiming timing;
    const auto outputs = RunStage(&stage, inputs, &timing, false);
    EnsureDir(argv[5]);
    for (size_t i = 0; i < outputs.size(); ++i) {
      std::ofstream file(std::string(argv[5]) + "/output" + std::to_string(i) + ".bin",
                         std::ios::out | std::ios::binary);
      file.write(reinterpret_cast<const char*>(TensorDataBytes(outputs[i].operator->())),
                 static_cast<std::streamsize>(TensorNBytes(outputs[i].operator->())));
      if (!file) throw std::runtime_error("binary probe output write failed");
      if (!naive && stage.output_live_input_indices[i] >= 0 &&
          outputs[i]->data != inputs[stage.output_live_input_indices[i]]->data)
        throw std::runtime_error("live-input output did not retain actual incoming buffer");
    }
    std::cout << "THESIS_NATIVE_CARRY_PROBE_COMPLETE mode=" << argv[6]
              << " carry_heads=" << carry_count << " outputs=" << outputs.size() << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
