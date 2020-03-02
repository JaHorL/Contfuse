#ifndef _TIMER_H
#define _TIMER_H

#include <time.h>

class Timer
{

public:

  clock_t time;

  Timer()
  {
  }

  ~Timer()
  {
  }


  void start()
  {
    time = (double) clock();
  }

  double stop()
  { 
    double diff_time = ((double) clock() - time) / CLOCKS_PER_SEC;
    time = (double)clock();
    return diff_time;
  }
};



#endif