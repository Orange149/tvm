// Independent scalar Cartesian oracle.  It does not use the DP half-vectors.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#define PICOJSON_USE_INT64
#include "picojson.h"

using Object = picojson::object;
using Array = picojson::array;
struct Cost {
  std::int64_t wall, run, boundary, ddr;
  std::array<std::int64_t,4> weights;
  std::array<std::int64_t,4> allocated{};
};
struct Program { bool cpu; int begin, end; std::vector<Cost> costs; };
struct Terminal { std::int64_t score; int graph; std::array<int, 4> allocation; bool gated; };
struct Allocation {
  std::array<int,4> values{};
  bool compatible{true}, single{true}, prefix{true};
  std::string shape;
};

std::map<std::string, Program> programs;
std::vector<std::vector<Program*>> graphs;
std::vector<std::vector<int>> options;
std::vector<int> masks, widths, pool, host;
std::map<int, std::vector<Allocation>> allocation_tables;
std::array<std::vector<Terminal>,3> best;
std::array<std::int64_t,3> counts{};
std::array<std::int64_t,2> gate_counts{};
std::map<std::string, std::vector<Terminal>> topology_strata, resource_strata;
std::string context_id;
int budget;
std::int64_t scale;

std::int64_t ceil_div(std::int64_t work,int capacity) {
  return work/capacity + (work%capacity!=0);
}

std::array<std::int64_t,4> allocate_run(const Cost& cost,int count) {
  std::int64_t denominator=0;
  for(int i=0;i<count;++i)denominator+=cost.weights[i];
  if(denominator<=0)throw std::runtime_error("invalid integer work weights");
  std::array<std::int64_t,4> result{},remainders{};
  std::array<int,4> order{0,1,2,3};
  std::int64_t allocated=0;
  for(int i=0;i<count;++i) {
    __int128 product=static_cast<__int128>(cost.run)*cost.weights[i];
    result[i]=static_cast<std::int64_t>(product/denominator);
    remainders[i]=static_cast<std::int64_t>(product%denominator);
    allocated+=result[i];
  }
  std::sort(order.begin(),order.begin()+count,[&](int a,int b){
    return remainders[a]!=remainders[b] ? remainders[a]>remainders[b] : a<b;
  });
  std::int64_t missing=cost.run-allocated;
  if(missing<0 || missing>count)throw std::runtime_error("work apportionment is not conservative");
  for(int i=0;i<missing;++i)++result[order[i]];
  return result;
}

std::vector<int> ints(const picojson::value& value) {
  std::vector<int> result;
  for (const auto& entry:value.get<Array>()) result.push_back(static_cast<int>(entry.get<double>()));
  return result;
}

void make_options(int position, int remaining, std::vector<int>& prefix) {
  if (!remaining) {
    options.push_back(prefix);
    int bits=0;
    for (int core:prefix) bits|=1<<core;
    masks.push_back(bits);
    widths.push_back(static_cast<int>(prefix.size()));
    return;
  }
  for (int i=position;i<=static_cast<int>(pool.size())-remaining;++i) {
    prefix.push_back(pool[i]); make_options(i+1,remaining-1,prefix);prefix.pop_back();
  }
}

bool less_terminal(const Terminal& a,const Terminal& b) {
  if (a.score!=b.score) return a.score<b.score;
  const auto& x=graphs[a.graph];const auto& y=graphs[b.graph];
  int cx=0,cy=0;
  for (std::size_t i=0;i<std::min(x.size(),y.size());++i) {
    const Program& px=*x[i];const Program& py=*y[i];
    // "cpu" precedes "vta" in the canonical Python tuple.
    if (px.cpu!=py.cpu) return px.cpu;
    if (px.begin!=py.begin) return px.begin<py.begin;
    if (px.end!=py.end) return px.end<py.end;
    int ax=px.cpu ? a.allocation[cx++] : -1;
    int ay=py.cpu ? b.allocation[cy++] : -1;
    int nx=px.cpu ? widths[ax] : 1;
    int ny=py.cpu ? widths[ay] : 1;
    if (nx!=ny) return nx<ny;
    const auto& mx=px.cpu ? options[ax] : host;
    const auto& my=py.cpu ? options[ay] : host;
    if (mx!=my) return mx<my;
  }
  if (x.size()!=y.size()) return x.size()<y.size();
  // All core/context fields are constant within this oracle invocation.
  return a.gated<b.gated;
}

void insert(std::vector<Terminal>& target,const Terminal& row,int k) {
  if (target.size()==static_cast<std::size_t>(k) && row.score>target.back().score) return;
  if (target.size()==static_cast<std::size_t>(k) && !less_terminal(row,target.back())) return;
  auto at=std::lower_bound(target.begin(),target.end(),row,less_terminal);
  target.insert(at,row);
  if (target.size()>static_cast<std::size_t>(k)) target.pop_back();
}

void make_allocations(int n,int position,Allocation& a,std::vector<Allocation>& target) {
  if (position<n) {
    for (int i=0;i<static_cast<int>(options.size());++i) {
      a.values[position]=i;make_allocations(n,position+1,a,target);
    }
    return;
  }
  a.compatible=a.single=a.prefix=true;
  a.shape=std::to_string(n)+":";
  for (int i=0;i<n;++i) {
    int option=a.values[i],prefix_mask=0;
    if (i) a.shape+=",";
    a.shape+=std::to_string(widths[option]);
    a.single &= widths[option]==1;
    for (int j=0;j<widths[option];++j) prefix_mask |= 1<<pool[j];
    a.prefix &= masks[option]==prefix_mask;
  }
  a.shape+=":";
  bool first=true;
  for (int i=0;i<n;++i) for (int j=i+1;j<n;++j) {
    int x=masks[a.values[i]],y=masks[a.values[j]],intersection=x&y;
    int relation=x==y ? 0 : !intersection ? 1 : (intersection==x || intersection==y) ? 2 : 3;
    if (!first) a.shape+=",";
    first=false;a.shape+=std::to_string(relation);
    a.compatible &= x==y || intersection==0;
  }
  target.push_back(a);
}

void evaluate(int graph_index,const Allocation& allocation) {
  std::array<std::int64_t,4> core{};
  std::array<std::int64_t,15> group_service{};
  std::array<std::int64_t,4> prefix{};
  std::int64_t cpu_max=0,vta=0,host_work=0,ddr=0;
  int cpu=0;
  for (auto* p:graphs[graph_index]) {
    if (!p->cpu) {
      const auto& c=p->costs[0];vta+=c.wall;host_work+=c.run+c.boundary;ddr+=c.ddr;
      continue;
    }
    int option=allocation.values[cpu++];const auto& c=p->costs[option];
    cpu_max=std::max(cpu_max,c.wall);ddr+=c.ddr;
    for (int k=0;k<widths[option];++k) core[options[option][k]]+=c.allocated[k];
    core[options[option][0]]+=c.boundary;
    group_service[option]+=c.wall;
    for (int width=widths[option];width<=4;++width) prefix[width-1]+=c.run;
  }
  std::int64_t sum=0,capacity=0;
  for (auto work:core) {sum+=work;capacity=std::max(capacity,work);}
  if (host.size()==1) capacity=std::max(capacity,core[host[0]]+host_work);
  else if (host.size()==4) capacity=std::max(capacity,ceil_div(sum+host_work,4));
  else {
    int bits=0;for(int id:host)bits|=1<<id;
    for(int subset=1;subset<16;++subset) {
      std::int64_t work=(bits&subset)==bits ? host_work : 0;
      int width=0;
      for(int id=0;id<4;++id)if(subset&(1<<id)){work+=core[id];++width;}
      capacity=std::max(capacity,ceil_div(work,width));
    }
  }
  auto service=std::max({cpu_max,vta,ddr});
  auto enhanced=std::max(service,capacity);
  auto serial=enhanced;
  for(auto work:group_service)serial=std::max(serial,work);
  auto old=std::max({service,ceil_div(sum+host_work,4),ceil_div(host_work,static_cast<int>(host.size()))});
  for(int width=1;width<=4;++width)old=std::max(old,ceil_div(prefix[width-1],width));
  for(int gate=0;gate<2;++gate) {
    if(gate && !allocation.compatible)continue;
    Terminal row{gate ? serial : enhanced,
                 graph_index,allocation.values,static_cast<bool>(gate)};
    ++counts[0];++gate_counts[gate];insert(best[0],row,budget);
    const std::string suffix=gate ? "/resource_group" : "/off";
    insert(topology_strata[std::to_string(graph_index)+suffix],row,2);
    insert(resource_strata[allocation.shape+suffix],row,2);
  }
  if(allocation.single) {
    ++counts[1];insert(best[1],Terminal{serial,
                              graph_index,allocation.values,true},budget);
  }
  if(allocation.prefix) {
    ++counts[2];insert(best[2],Terminal{old,
                              graph_index,allocation.values,false},budget);
  }
}

picojson::value terminal_json(const Terminal& row) {
  int count=0;for(auto* p:graphs[row.graph])count+=p->cpu;
  Array allocation;
  for(int i=0;i<count;++i)allocation.emplace_back(static_cast<double>(row.allocation[i]));
  return picojson::value(Array{picojson::value(static_cast<int64_t>(row.score)),
      picojson::value(static_cast<double>(row.graph)),picojson::value(allocation),
      picojson::value(context_id),picojson::value(row.gated ? "resource_group" : "off")});
}

picojson::value rows_json(const std::vector<Terminal>& rows) {
  Array result;for(const auto& row:rows)result.push_back(terminal_json(row));
  return picojson::value(result);
}

picojson::value strata_json(const std::map<std::string,std::vector<Terminal>>& strata) {
  Object result;for(const auto& entry:strata)result[entry.first]=rows_json(entry.second);
  return picojson::value(result);
}

int main() {
  try {
    picojson::value input;std::cin>>input;
    if(!picojson::get_last_error().empty())throw std::runtime_error(picojson::get_last_error());
    auto& root=input.get<Object>();auto& context=root.at("context").get<Object>();
    context_id=context.at("id").get<std::string>();
    pool=ints(context.at("compute_pool"));host=ints(context.at("host_mask"));
    budget=static_cast<int>(root.at("k").get<double>());scale=root.at("score_scale").get<int64_t>();
    if(scale!=1000000)throw std::runtime_error("oracle expects conserved integer nanoseconds");
    std::vector<int> scratch;
    for(int n=1;n<=static_cast<int>(pool.size());++n)make_options(0,n,scratch);
    for(auto& entry:root.at("programs").get<Object>()) {
      auto& value=entry.second.get<Object>();auto& program=programs[entry.first];
      program.cpu=value.at("cpu").get<bool>();
      program.begin=static_cast<int>(value.at("begin").get<double>());
      program.end=static_cast<int>(value.at("end").get<double>());
      for(auto& raw:value.at("costs").get<Array>()) {
        auto& row=raw.get<Array>();program.costs.push_back({row[0].get<int64_t>(),row[1].get<int64_t>(),
            row[2].get<int64_t>(),row[3].get<int64_t>(), {row[4].get<int64_t>(),row[5].get<int64_t>(),
                                                     row[6].get<int64_t>(),row[7].get<int64_t>()}});
        if(program.cpu) {
          const int option=static_cast<int>(program.costs.size())-1;
          program.costs.back().allocated=allocate_run(program.costs.back(),widths[option]);
        }
      }
    }
    for(auto& raw:root.at("topologies").get<Array>()) {
      std::vector<Program*> topology;
      for(auto& name:raw.get<Array>())topology.push_back(&programs.at(name.get<std::string>()));
      graphs.push_back(topology);
    }
    for(int graph=0;graph<static_cast<int>(graphs.size());++graph) {
      int n=0;for(auto* p:graphs[graph])n+=p->cpu;
      if(n<2 || n>4)throw std::runtime_error("unsupported CPU stage count");
      if(!allocation_tables.count(n)) {
        Allocation scratch;make_allocations(n,0,scratch,allocation_tables[n]);
      }
      for(const auto& allocation:allocation_tables[n])evaluate(graph,allocation);
    }
    Object result,count,ranks,gates;
    std::array<std::string,3> names{"enhanced","single_core","prefix"};
    for(int i=0;i<3;++i){count[names[i]]=picojson::value(static_cast<double>(counts[i]));ranks[names[i]]=rows_json(best[i]);}
    gates["off"]=picojson::value(static_cast<double>(gate_counts[0]));
    gates["resource_group"]=picojson::value(static_cast<double>(gate_counts[1]));
    result["counts"]=picojson::value(count);result["rankings"]=picojson::value(ranks);
    result["gate_counts"]=picojson::value(gates);
    result["topology_strata"]=strata_json(topology_strata);
    result["resource_strata"]=strata_json(resource_strata);
    std::cout<<picojson::value(result).serialize()<<"\n";
  } catch(const std::exception& error) {std::cerr<<error.what()<<"\n";return 2;}
}
