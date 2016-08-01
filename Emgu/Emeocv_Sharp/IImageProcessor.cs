using Emgu.CV;
using Emgu.CV.Structure;
using System.Collections.Generic;

namespace Emeocv_Sharp
{
    public interface IImageProcessor
    {
        void setInput(Image<Bgr, byte> img);
        void process();
        List<Image<Gray, byte>> getOutput();

        void debugWindow(bool bval = true);
        void debugSkew(bool bval = true);
        void debugEdges(bool bval = true);
        void debugDigits(bool bval = true);
        void showImage();
    }
}