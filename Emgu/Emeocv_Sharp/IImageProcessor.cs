using System.Collections.Generic;

namespace Emeocv_Sharp
{
    public interface IImageProcessor
    {
        void setInput(Mat img);
        void process();
        List<Mat> getOutput();

        void debugWindow(bool bval = true);
        void debugSkew(bool bval = true);
        void debugEdges(bool bval = true);
        void debugDigits(bool bval = true);
        void showImage();
    }
}