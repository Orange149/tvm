// Write only to an already running experiment's FIFO; never create an IPC path.
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <signal.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
int main(int argc, char** argv) {
  if (argc != 3) { std::fprintf(stderr, "usage: command FIFO 'release ID|abort'\n"); return 2; }
  const std::string command(argv[2]);
  if (command != "abort") {
    const std::string prefix = "release ";
    const auto id = command.size() >= prefix.size() ? command.substr(prefix.size()) : "";
    if (command.rfind(prefix, 0) != 0 || id.empty() || id.size() > 80 ||
        id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_") != std::string::npos) {
      std::fprintf(stderr, "invalid lifecycle command\n"); return 2;
    }
  }
  signal(SIGPIPE, SIG_IGN);
  const int descriptor = open(argv[1], O_WRONLY | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC);
  if (descriptor < 0) { std::fprintf(stderr, "cannot open live lifecycle FIFO: %s\n", std::strerror(errno)); return 1; }
  struct stat info{};
  if (fstat(descriptor, &info) != 0 || !S_ISFIFO(info.st_mode)) {
    close(descriptor); std::fprintf(stderr, "lifecycle command target is not a FIFO\n"); return 1;
  }
  const std::string bytes = command + "\n";
  ssize_t count;
  do { count = write(descriptor, bytes.data(), bytes.size()); } while (count < 0 && errno == EINTR);
  close(descriptor);
  if (count != static_cast<ssize_t>(bytes.size())) { std::fprintf(stderr, "lifecycle command write failed\n"); return 1; }
  std::printf("LIFECYCLE_COMMAND_WRITTEN: verify command_received/released in native trace\n");
  return 0;
}
