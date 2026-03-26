#include <ESP32Video.h>
#include <Ressources/CodePage437_8x8.h>

const int outputPin = 26;

CompositeColorDAC videodisplay;

void setup()
{
  videodisplay.init(CompMode::MODENTSCColor240P, 26, false);

  videodisplay.fillCircle(25 + 55, 50, 1 + 50 / 4, videodisplay.RGB(96, 0, 0));
  videodisplay.fillCircle(25 + 55+30, 50, 1 + 50 / 4, videodisplay.RGB(0, 96, 0));
  videodisplay.fillCircle(25 + 55+60, 50, 1 + 50 / 4, videodisplay.RGB(0, 0, 96));

  videodisplay.fillCircle(100 + 25 + 55, 50, 1 + 50 / 4, videodisplay.RGB(255, 0, 0));
  videodisplay.fillCircle(100 + 25 + 55+30, 50, 1 + 50 / 4, videodisplay.RGB(0, 255, 0));
  videodisplay.fillCircle(100 + 25 + 55+60, 50, 1 + 50 / 4, videodisplay.RGB(0, 0, 255));

  videodisplay.fillCircle(200 + 25 + 55, 50, 1 + 50 / 4, videodisplay.RGB(96, 96, 0));
  videodisplay.fillCircle(200 + 25 + 55+30, 50, 1 + 50 / 4, videodisplay.RGB(0, 96, 96));
  videodisplay.fillCircle(200 + 25 + 55+60, 50, 1 + 50 / 4, videodisplay.RGB(96, 0, 96));

  videodisplay.fillCircle(300 + 25 + 55, 50, 1 + 50 / 4, videodisplay.RGB(255, 255, 0));
  videodisplay.fillCircle(300 + 25 + 55+30, 50, 1 + 50 / 4, videodisplay.RGB(0, 255, 255));
  videodisplay.fillCircle(300 + 25 + 55+60, 50, 1 + 50 / 4, videodisplay.RGB(255, 0, 255));



  videodisplay.setFont(CodePage437_8x8);
  videodisplay.fillRect(8+30-4, 88-2, (2*255)+5+8, 40+4+4, videodisplay.RGB(127,127,127));
  videodisplay.fillRect(8+30, 88, (2*255)+5, 40+4, videodisplay.RGB(0,0,0));
  videodisplay.setTextColor(videodisplay.RGB(192,192,192));
  for(int x = 0; x < 256*2; x+=2)
  {
    videodisplay.fillRect(8+x + 32, 90, 2, 40, videodisplay.RGB(x/2,x/2,x/2));
    if(x % 32 == 0)
    {
      videodisplay.fillRect(8+x + 32, 85, 4, 4, videodisplay.RGB(255,255,255));
      videodisplay.setCursor(8+x + 32 - 4, 78 - 4);
      videodisplay.print(x/2,HEX);
    }
  }
}

void loop()
{
}
