// Independent scalar enumeration used only for local algorithm verification.
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "picojson.h"

using Obj = picojson::object;
using Arr = picojson::array;
struct Cost { double wall,run,boundary; };
struct Program { bool cpu; std::vector<Cost> costs; std::vector<std::string> keys; };
struct Candidate { long long value; int topology; std::array<int,4> allocation; bool gated; };
const std::vector<std::string> methods = {"single_core","fixed_groups_serial","fixed_groups_open",
    "total_capacity","existing","subset_capacity","legacy_independent"};
std::vector<std::vector<Program*>> graphs;
std::vector<int> masks,widths;
std::array<std::vector<Candidate>,7> best;
std::array<long long,7> counts{};
int hostmask; bool dedicated;

bool less_candidate(const Candidate& a,const Candidate& b) {
  if(a.value!=b.value) return a.value<b.value;
  const auto& x=graphs[a.topology]; const auto& y=graphs[b.topology];
  int cx=0,cy=0;
  for(size_t i=0;i<std::min(x.size(),y.size());++i) {
    const auto& sx=x[i]->keys[x[i]->cpu ? a.allocation[cx++] : 0];
    const auto& sy=y[i]->keys[y[i]->cpu ? b.allocation[cy++] : 0];
    if(sx!=sy) return sx<sy;
  }
  if(x.size()!=y.size()) return x.size()<y.size();
  return a.gated<b.gated;
}

void evaluate(int graph,const std::array<int,4>& allocation) {
  std::array<int,4> cm{}; std::array<double,4> wall{},run{},work{};
  int n=0; double vta=0,host=0,cpu=0,slow=0;
  for(auto* p:graphs[graph]) {
    if(p->cpu) {
      int o=allocation[n]; auto c=p->costs[o];
      cm[n]=masks[o];wall[n]=c.wall;run[n]=c.run;work[n]=c.run+c.boundary;
      slow=std::max(slow,c.wall);cpu+=work[n];++n;
    } else { vta+=p->costs[0].wall;host+=p->costs[0].run+p->costs[0].boundary; }
  }
  double service=std::max(slow,vta),total=std::max(service,(cpu+host)/4);
  if(dedicated) total=std::max({total,cpu/3,host});
  double subset=service;
  for(int a=1;a<16;++a) {
    double sum=(hostmask&a)==hostmask ? host : 0;
    for(int j=0;j<n;++j) if((cm[j]&a)==cm[j]) sum+=work[j];
    subset=std::max(subset,sum/__builtin_popcount(static_cast<unsigned>(a)));
  }
  bool fixed=true,single=true,prefix=true;
  double device=total,existing=dedicated ? subset : std::max(service,(cpu+host)/4);
  for(int j=0;j<n;++j) {
    single &= widths[allocation[j]]==1;
    prefix &= cm[j]==(1<<widths[allocation[j]])-1;
    double sum=0;
    for(int l=0;l<n;++l) {
      if(cm[j]==cm[l]) sum+=wall[l];
      if(cm[j]!=cm[l] && (cm[j]&cm[l])) fixed=false;
    }
    device=std::max(device,sum);
  }
  if(!dedicated) for(int k=1;k<=4;++k) {
    double sum=0;
    for(int j=0;j<n;++j) if(widths[allocation[j]]<=k) sum+=run[j];
    existing=std::max(existing,sum/k);
  }
  const std::array<double,7> scores{device,device,device,total,existing,subset,service};
  const std::array<bool,7> eligible{single,fixed,fixed,true,prefix,true,true};
  for(int m=0;m<7;++m) {
    if(!eligible[m]) continue;
    ++counts[m];
    Candidate value{static_cast<long long>(std::nearbyint(scores[m]*1e9)),graph,allocation,m==1};
    auto& pool=best[m];
    if(pool.size()==20 && !less_candidate(value,pool.back())) continue;
    pool.insert(std::lower_bound(pool.begin(),pool.end(),value,less_candidate),value);
    if(pool.size()>20) pool.pop_back();
  }
}

void enumerate(int graph,int n,int i,std::array<int,4>& allocation) {
  if(i==n) { evaluate(graph,allocation);return; }
  for(int o=0;o<static_cast<int>(masks.size());++o) { allocation[i]=o;enumerate(graph,n,i+1,allocation); }
}

int main() {
  picojson::value input; std::cin>>input;
  if(!picojson::get_last_error().empty()) return 2;
  auto& root=input.get<Obj>();hostmask=static_cast<int>(root["host_mask"].get<double>());
  dedicated=root["dedicated"].get<bool>();
  for(auto& o:root["options"].get<Arr>()) {
    masks.push_back(static_cast<int>(o.get<Obj>()["mask"].get<double>()));
    widths.push_back(static_cast<int>(o.get<Obj>()["threads"].get<double>()));
  }
  std::map<std::string,Program> programs;
  for(auto& item:root["programs"].get<Obj>()) {
    auto& p=programs[item.first];auto& v=item.second.get<Obj>();p.cpu=v["cpu"].get<bool>();
    for(auto& c:v["costs"].get<Arr>()) {
      auto& a=c.get<Arr>();p.costs.push_back({a[0].get<double>(),a[1].get<double>(),a[2].get<double>()});
    }
    for(auto& key:v["keys"].get<Arr>()) p.keys.push_back(key.get<std::string>());
  }
  for(auto& value:root["topologies"].get<Arr>()) {
    std::vector<Program*> topology;
    for(auto& name:value.get<Arr>()) topology.push_back(&programs.at(name.get<std::string>()));
    graphs.push_back(topology);
  }
  for(int g=0;g<static_cast<int>(graphs.size());++g) {
    int n=0;for(auto* p:graphs[g]) n+=p->cpu;
    std::array<int,4> allocation{};enumerate(g,n,0,allocation);
  }
  Obj output,ranking,count;
  for(int m=0;m<7;++m) {
    count[methods[m]]=picojson::value(static_cast<double>(counts[m]));
    Arr rows;
    for(auto& c:best[m]) {
      std::string key;int cpu=0;
      for(auto* p:graphs[c.topology]) { if(!key.empty()) key+="__";key+=p->keys[p->cpu ? c.allocation[cpu++] : 0]; }
      key+=c.gated ? "__gate_resource_group" : "__gate_off";
      Obj row;row["id"]=picojson::value(key);row["score_ms"]=picojson::value(c.value/1e9);rows.push_back(picojson::value(row));
    }
    ranking[methods[m]]=picojson::value(rows);
  }
  count["topologies"]=picojson::value(static_cast<double>(graphs.size()));
  output["counts"]=picojson::value(count);output["rankings"]=picojson::value(ranking);
  std::cout<<picojson::value(output).serialize()<<std::endl;
}
