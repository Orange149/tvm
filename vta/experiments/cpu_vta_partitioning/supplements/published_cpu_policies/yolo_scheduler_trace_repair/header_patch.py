"""Small, checked patch over the already frozen optional observer source."""
def once(text, old, new):
    if text.count(old)!=1: raise ValueError('frozen source hook changed')
    return text.replace(old,new,1)


def patched_header(original):
    text=once(original,'#include <algorithm>','#include "ring_drain.h"\n#include <algorithm>')
    text=once(text,'Count before,after;std::vector<unsigned char> bytes;',
              'Count before,after;scheduler_ring::Retained retained;')
    text=once(text,' void Write(){',r'''
 void DrainTask(Task& t) {
   auto* m=static_cast<perf_event_mmap_page*>(t.mapped);
   U offset=m->data_offset?m->data_offset:t.page;
   t.data_size=m->data_size?m->data_size:t.page*32;
   if(offset<t.page || !t.data_size || offset+t.data_size>t.map_bytes)
     throw std::runtime_error("invalid scheduler ring layout");
   t.tail=__atomic_load_n(&m->data_tail,__ATOMIC_RELAXED);
   scheduler_ring::Drain(t.retained,static_cast<unsigned char*>(t.mapped)+offset,
     t.data_size,t.tail,
     [&](){return __atomic_load_n(&m->data_head,__ATOMIC_ACQUIRE);},
     [](){return Clock();},
     [&](U h){__atomic_store_n(&m->data_tail,h,__ATOMIC_RELEASE);});
   t.head=t.retained.cursor;
 }
 void Write(){''')
    text=once(text,r'\"periodic_sampling\":false,\"data_pages\":32,',
        r'\"periodic_sampling\":false,\"data_pages\":32,\"ring_trace_revision\":2,\"drain_policy\":\"collector_completion\",')
    text=once(text,'    <<"\\\"completion_start_ns\\\":"<<completion_start_',
              '    <<"\\\"retention_end_ns\\\":"<<Clock()<<",\\\"completion_start_ns\\\":"<<completion_start_')
    text=once(text,'    for(unsigned char b:t.bytes)o<<hex[b>>4]<<hex[b&15];o<<"\\\"}";',r'''
    for(unsigned char b:t.retained.bytes)o<<hex[b>>4]<<hex[b&15];
    o<<"\",\"retained_bytes\":"<<t.retained.bytes.size()<<",\"drain_chunks\":[";
    for(size_t j=0;j<t.retained.chunks.size();++j){if(j)o<<",";const auto& c=t.retained.chunks[j];
      o<<"{\"tail\":"<<c.tail<<",\"head\":"<<c.head<<",\"head_after_copy\":"<<c.head_after_copy
       <<",\"head_after_publish\":"<<c.head_after_publish
       <<",\"begin_ns\":"<<c.begin_ns<<",\"end_ns\":"<<c.end_ns<<",\"offset\":"<<c.offset<<"}";}
    o<<"]}";''')
    start=text.index('    for(auto& t:tasks_){auto* m=static_cast<perf_event_mmap_page*>')
    end=text.index('\n   }\n   Close();Write();',start)
    text=text[:start]+'    for(auto& t:tasks_)DrainTask(t);'+text[end:]
    text=once(text,' void Stop()noexcept{',r'''
 void Checkpoint()noexcept{
   if(!enabled()||!started_||stopped_||!error_.empty())return;
   try{for(auto& t:tasks_)DrainTask(t);}
   catch(const std::exception& e){error_=e.what();}
   catch(...){error_="unknown scheduler checkpoint failure";}
 }
 void Stop()noexcept{''')
    # Even after a failed checkpoint, disable every event before close/write.
    text=once(text,'   Close();Write();',
        '   if(!error_.empty())for(auto& t:tasks_){if(t.event>=0)ioctl(t.event,PERF_EVENT_IOC_DISABLE,0);if(t.counter>=0)ioctl(t.counter,PERF_EVENT_IOC_DISABLE,0);}\n   Close();Write();')
    return text
