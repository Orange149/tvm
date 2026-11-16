// Optional observation only. Linux TASKSTATS per-TID counters, never TGID sums.
#pragma once
#include <linux/genetlink.h>
#include <linux/netlink.h>
#include <linux/taskstats.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace scheduler_gap {
struct Counters { uint64_t delay=0, runtime=0, count=0; uint16_t version=0; };
class Client {
  int fd_=-1, family_=0; uint32_t sequence_=0;
  static size_t align(size_t n) {return (n+3)&~size_t(3);}
  std::vector<char> request(int type,int command,int attr,const void* data,size_t size) {
    std::vector<char> b(NLMSG_HDRLEN+GENL_HDRLEN+align(NLA_HDRLEN+size),0);
    auto* n=reinterpret_cast<nlmsghdr*>(b.data()); n->nlmsg_len=b.size();
    n->nlmsg_type=type;n->nlmsg_flags=NLM_F_REQUEST;n->nlmsg_seq=++sequence_;
    auto* g=reinterpret_cast<genlmsghdr*>(b.data()+NLMSG_HDRLEN);g->cmd=command;g->version=1;
    auto* a=reinterpret_cast<nlattr*>(b.data()+NLMSG_HDRLEN+GENL_HDRLEN);
    a->nla_type=attr;a->nla_len=NLA_HDRLEN+size;memcpy(reinterpret_cast<char*>(a)+NLA_HDRLEN,data,size);
    sockaddr_nl peer{};peer.nl_family=AF_NETLINK;
    if(sendto(fd_,b.data(),b.size(),0,reinterpret_cast<sockaddr*>(&peer),sizeof(peer))!=ssize_t(b.size()))
      throw std::runtime_error("TASKSTATS send: "+std::string(strerror(errno)));
    b.resize(32768); ssize_t got;
    do {got=recv(fd_,b.data(),b.size(),0);}while(got<0&&errno==EINTR);
    if(got<ssize_t(NLMSG_HDRLEN))throw std::runtime_error("TASKSTATS receive: "+std::string(strerror(errno)));
    n=reinterpret_cast<nlmsghdr*>(b.data());
    if(n->nlmsg_seq!=sequence_||n->nlmsg_len>size_t(got)||n->nlmsg_len<NLMSG_HDRLEN)
      throw std::runtime_error("TASKSTATS message identity/length mismatch");
    if(n->nlmsg_type==NLMSG_ERROR) {
      if(n->nlmsg_len<NLMSG_HDRLEN+sizeof(nlmsgerr))throw std::runtime_error("short netlink error");
      const auto e=reinterpret_cast<nlmsgerr*>(b.data()+NLMSG_HDRLEN)->error;
      throw std::runtime_error("TASKSTATS netlink error: "+std::string(strerror(-e)));
    }
    b.resize(n->nlmsg_len);return b;
  }
  template<class F> static void attrs(const char* b,size_t size,F f) {
    size_t pos=0;
    while(pos+NLA_HDRLEN<=size) {
      const auto* a=reinterpret_cast<const nlattr*>(b+pos);
      if(a->nla_len<NLA_HDRLEN||pos+a->nla_len>size)throw std::runtime_error("invalid netlink attribute");
      f(a->nla_type&NLA_TYPE_MASK,b+pos+NLA_HDRLEN,a->nla_len-NLA_HDRLEN);
      pos+=align(a->nla_len);
    }
  }
 public:
  Client() {
    fd_=socket(AF_NETLINK,SOCK_RAW|SOCK_CLOEXEC,NETLINK_GENERIC);
    if(fd_<0)throw std::runtime_error("TASKSTATS socket failed");
    try {
      timeval timeout{0,250000};setsockopt(fd_,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));
      sockaddr_nl local{};local.nl_family=AF_NETLINK;
      if(bind(fd_,reinterpret_cast<sockaddr*>(&local),sizeof(local)))throw std::runtime_error("TASKSTATS bind failed");
      const char name[]=TASKSTATS_GENL_NAME;
      auto b=request(GENL_ID_CTRL,CTRL_CMD_GETFAMILY,CTRL_ATTR_FAMILY_NAME,name,sizeof(name));
      attrs(b.data()+NLMSG_HDRLEN+GENL_HDRLEN,b.size()-NLMSG_HDRLEN-GENL_HDRLEN,
        [&](int t,const char* p,size_t n){if(t==CTRL_ATTR_FAMILY_ID&&n==2){uint16_t v;memcpy(&v,p,2);family_=v;}});
      if(!family_)throw std::runtime_error("TASKSTATS family missing");
    }catch(...){close(fd_);fd_=-1;throw;}
  }
  Client(const Client&)=delete;
  ~Client(){if(fd_>=0)close(fd_);}
  Counters read(uint32_t tid) {
    auto b=request(family_,TASKSTATS_CMD_GET,TASKSTATS_CMD_ATTR_PID,&tid,sizeof(tid));
    bool found=false;uint32_t actual=0;Counters result;
    attrs(b.data()+NLMSG_HDRLEN+GENL_HDRLEN,b.size()-NLMSG_HDRLEN-GENL_HDRLEN,
      [&](int t,const char* p,size_t n){
        if(t!=TASKSTATS_TYPE_AGGR_PID)return;
        attrs(p,n,[&](int a,const char* q,size_t len){
          if(a==TASKSTATS_TYPE_PID&&len==4)memcpy(&actual,q,4);
          if(a==TASKSTATS_TYPE_STATS) {
            if(len<offsetof(taskstats,cpu_run_virtual_total)+8)throw std::runtime_error("short TASKSTATS payload");
            taskstats s{};memcpy(&s,q,len<sizeof(s)?len:sizeof(s));
            result={s.cpu_delay_total,s.cpu_run_virtual_total,s.cpu_count,s.version};found=true;
          }
        });
      });
    if(!found||actual!=tid)throw std::runtime_error("TASKSTATS replied for wrong TID");
    return result;
  }
};
}  // namespace scheduler_gap
