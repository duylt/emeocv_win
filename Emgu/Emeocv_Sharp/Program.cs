using OpenCV.Net;
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
            var image = CV.LoadImageM("D:\\meter.jpg", LoadImageFlags.Color);
            Mat img_cvt = new Mat(image.Rows,image.Cols,image.Depth,image.Channels);
            CV.CvtColor(image, img_cvt, ColorConversion.BayerBG2Gray);
            CV.SaveImage("D:\\meter2.jpg", img_cvt);
        }
    }
}
