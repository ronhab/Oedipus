#ifdef __APPLE__
#include<Availability.h>
#include<sys/cdefs.h>
#undef __OSX_AVAILABLE_STARTING
#define __OSX_AVAILABLE_STARTING(_mac, _iphone)
#undef __OSX_AVAILABLE_BUT_DEPRECATED
#define __OSX_AVAILABLE_BUT_DEPRECATED(_osxIntro, _osxDep, _iosIntro, _iosDep)
#undef __OSX_AVAILABLE_BUT_DEPRECATED_MSG
#define __OSX_AVAILABLE_BUT_DEPRECATED_MSG(_osxIntro, _osxDep, _iosIntro, _iosDep, _msg)
#undef __OSX_DEPRECATED
#define __OSX_DEPRECATED(_start, _dep, _msg)
#undef __IOS_DEPRECATED
#define __IOS_DEPRECATED(_start, _dep, _msg)
#undef __TVOS_DEPRECATED
#define __TVOS_DEPRECATED(_start, _dep, _msg)
#undef __WATCHOS_DEPRECATED
#define __WATCHOS_DEPRECATED(_start, _dep, _msg)
#undef __OSX_AVAILABLE
#define __OSX_AVAILABLE(_arg)
#undef __IOS_AVAILABLE
#define __IOS_AVAILABLE(_arg)
#undef __TVOS_AVAILABLE
#define __TVOS_AVAILABLE(_arg)
#undef __WATCHOS_AVAILABLE
#define __WATCHOS_AVAILABLE(_arg)
#undef __OSX_PROHIBITED
#define __OSX_PROHIBITED
#undef __IOS_PROHIBITED
#define __IOS_PROHIBITED
#undef __TVOS_PROHIBITED
#define __TVOS_PROHIBITED
#undef __WATCHOS_PROHIBITED
#define __WATCHOS_PROHIBITED
#undef __OS_AVAILABILITY_MSG
#define __OS_AVAILABILITY_MSG(a,b,c)
#undef __BLOCKS__
#undef _Nullable
#define _Nullable
#undef _Nonnull
#define _Nonnull
#undef __swift_unavailable
#define __swift_unavailable(_msg)
#endif

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void fac(int n) {
  int s=1;
  int i;
  for (i=2; i<=n; i++) {
    s *= i;
  }
  printf("fac(%i)=%i\n",n,s);
}

void fib(int n) {
  int a=0;
  int b=1;
  int s=1;

  int i;
  for (i=1; i<n; i++) {
    s=a+b;
    a=b;
    b=s;
  }
  printf("fib(%i)=%i\n",n,s);
}

int main (int argc, char** argv) {
  fac(1);
  fib(1);
  fac(5);
  fib(5);
  fac(10);
  fib(10);
  return 0;
}
