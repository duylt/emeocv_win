using Emgu.CV;
using Emgu.CV.Structure;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Emeocv_Sharp
{
    class Program
    {
        static void Main(string[] args)
        {
            using (ImageProcessor processor = new ImageProcessor())
            {
                processor.setInput(new Image<Bgr, byte>("J:\\meter.jpg"));
                processor.process();
            }
        }
    }
}
