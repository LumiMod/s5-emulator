#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include "teakra/disassembler.h"
int main(int argc,char**argv) {
 if(argc!=5)return 2;
 std::ifstream f(argv[1],std::ios::binary);std::vector<unsigned char>b((std::istreambuf_iterator<char>(f)),{});
 unsigned first=std::strtoul(argv[2],nullptr,0),end=std::strtoul(argv[3],nullptr,0),group=std::strtoul(argv[4],nullptr,0);
 if(!group||first>end||end>b.size()/2||b.size()%(group*2))return 3;
 auto rd=[&](unsigned i){unsigned j=(i/group)*group+group-1-i%group;return unsigned(b[2*j])|(unsigned(b[2*j+1])<<8);};
 for(unsigned i=first;i<end;){unsigned x=rd(i),e=0;bool more=Teakra::Disassembler::NeedExpansion(x);
  if(more){if(i+1>=b.size()/2)return 4;e=rd(i+1);}
  std::printf("word %04x: %04x %s %s\n",i,x,more?"+word":"     ",Teakra::Disassembler::Do(x,e).c_str());i+=more?2:1;
 }
}
