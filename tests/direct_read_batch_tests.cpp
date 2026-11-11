#include "storage/direct_read_batch.hpp"
#include "storage/kv_checksum.hpp"
#include "storage/file_store.hpp"
#include <filesystem>
#include <fcntl.h>
#include <future>
#include <iostream>
#include <memory>
#include <vector>
#include <cstdlib>
#include <cstring>
using namespace hillserve;
static void check(bool ok){if(!ok)throw std::runtime_error("bounded reader/checksum contract");}
template<class F>static void rejects(F f){bool rejected=false;try{f();}catch(const std::exception&){rejected=true;}check(rejected);}
int main(){const auto path=std::filesystem::temp_directory_path()/("hs-read-batch-"+std::to_string(getpid()));int fd=-1;
 try{
  std::vector<unsigned char> bytes(4*65536);for(std::size_t i=0;i<bytes.size();++i)bytes[i]=(i/4096*31+i*17)%251;
  std::array<const unsigned char*,4> rows{};for(std::size_t i=0;i<4;++i)rows[i]=bytes.data()+i*65536;
  for(auto length:{0,1,17,4096,65536})for(std::size_t n=1;n<=4;++n){const auto got=storage::checksum_four(rows,length,n);for(std::size_t i=0;i<n;++i)check(got[i]==checksum(rows[i],length));}
  int writer=open(path.c_str(),O_CREAT|O_EXCL|O_WRONLY,0600);check(writer>=0);check(write(writer,bytes.data(),bytes.size())==ssize_t(bytes.size()));check(fsync(writer)==0);close(writer);
#ifdef O_DIRECT
  fd=open(path.c_str(),O_RDONLY|O_DIRECT|O_CLOEXEC);
#else
  // Mac portable test is explicitly buffered; Linux test exercises O_DIRECT.
  fd=open(path.c_str(),O_RDONLY|O_CLOEXEC);
#endif
  check(fd>=0);storage::DirectReadBatch reader(4);
  auto task=[&](int which){
   void* raw=nullptr;check(posix_memalign(&raw,4096,4*65536)==0);std::unique_ptr<void,decltype(&free)> buffer(raw,&free);
   std::array<storage::DirectReadBatch::Request,4> requests{};
   for(std::size_t i=0;i<4;++i)requests[i]={fd,static_cast<unsigned char*>(raw)+i*65536,65536,((i+which)%4)*65536};
   for(int round=0;round<32;++round){const auto n=std::size_t(round%4+1);reader.read_exact(requests,n);for(std::size_t i=0;i<n;++i)check(std::memcmp(requests[i].data,bytes.data()+requests[i].offset,65536)==0);}
   const auto saved=requests[3].offset;requests[3].offset=bytes.size();rejects([&]{reader.read_exact(requests,4);});requests[3].offset=saved;
   // Failure drains every worker and leaves the pool usable for a new call.
   reader.read_exact(requests,4);for(const auto& r:requests)check(std::memcmp(r.data,bytes.data()+r.offset,r.bytes)==0);
   rejects([&]{reader.read_exact(requests,5);});requests[0].offset=1;rejects([&]{reader.read_exact(requests,1);});
  };
  auto a=std::async(std::launch::async,[&]{task(0);});auto b=std::async(std::launch::async,[&]{task(1);});a.get();b.get();
  rejects([]{storage::DirectReadBatch bad(0);});rejects([]{storage::DirectReadBatch bad(5);});
  close(fd);fd=-1;std::filesystem::remove(path);
  std::cout<<"bounded four-read pool, concurrent callers, short IO drain/reuse, and unchanged FNV checksums passed\n";return 0;
 }catch(const std::exception& e){if(fd>=0)close(fd);std::filesystem::remove(path);std::cerr<<e.what()<<'\n';return 1;}
}
