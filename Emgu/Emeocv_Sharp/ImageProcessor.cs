using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using EmeocvSharp;
using Emgu.CV;
using System.Drawing;
using Emgu.CV.Util;
using Emgu.CV.Structure;

namespace Emeocv_Sharp
{
    public class ImageProcessor : IImageProcessor,IDisposable
    {
        private Config _config { get; set; }
        private Image<Bgr, byte> _img { get; set; }
        private Image<Gray, byte> _imgGray { get; set; }
        private List<Image<Gray, byte>> _digits { get; set; }
        private bool _debugWindow { get; set; }
        private bool _debugSkew { get; set; }
        private bool _debugEdges { get; set; }
        private bool _debugDigits { get; set; }

        public ImageProcessor()
        {
            this._config = new Config();
            debugWindow(false);
            debugSkew(false);
            debugDigits(false);
            debugEdges(false);
            _digits = new List<Image<Gray, byte>>();
        }
        public void debugDigits(bool bval = true)
        {
            this._debugDigits = bval;
        }

        public void debugEdges(bool bval = true)
        {
            this._debugEdges = bval;
        }

        public void debugSkew(bool bval = true)
        {
            this._debugSkew = bval;
        }

        public void debugWindow(bool bval = true)
        {
            this._debugWindow = bval;
        }

        public void process()
        {
            findCounterDigits();
        }

        public void setInput(Image<Bgr, byte> img)
        {
            this._img = img;
            this._imgGray = img.Convert<Gray, byte>();
        }

        public void showImage()
        {
            Console.ReadLine();
        }

        public List<Image<Gray, byte>> getOutput()
        {
            return _digits;
        }

        private void rotate(double rotationDegrees)
        {
            UMat mapMatrix = new UMat();
            Image<Gray, byte> img_rotated = this._imgGray;
             Emgu.CV.CvInvoke.GetRotationMatrix2D(new PointF(this._imgGray.Cols / 2, this._imgGray.Rows / 2), rotationDegrees, 1, mapMatrix);

            Emgu.CV.CvInvoke.WarpAffine(this._imgGray, img_rotated, mapMatrix,this._img.Size);
            this._imgGray = img_rotated;
            //if (this._debugWindow)
            //{
            //    Emgu.CV.CvInvoke.WarpAffine(this._img, img_rotated, mapMatrix, this._img.Size);
            //    this._img = img_rotated;
            //}
        }

        private void findCounterDigits()
        {
            UMat edges = cannyEdges();
            if (this._debugEdges)
            {
                //imshow("edges", edges);
            }

            UMat img_ret = edges.Clone();
            
            List<Rectangle> boundingBoxes = new List<Rectangle>();
            VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
            VectorOfVectorOfPoint filteredContours = new VectorOfVectorOfPoint();
            
            //Find contours
            CvInvoke.FindContours(edges, contours, null, Emgu.CV.CvEnum.RetrType.List, Emgu.CV.CvEnum.ChainApproxMethod.ChainApproxNone);
         
            filterContours(contours, boundingBoxes, filteredContours);

            //Draw contours
            for(var index = 0; index< contours.Size;index++)
            {
                Rectangle bounds = CvInvoke.BoundingRectangle(contours[index]);
                this._img.Draw(bounds, new Bgr(Color.Green), 1);
            }
            this._imgGray.Save("J:\\contouredImage.jpg");


            //// find bounding boxes that are aligned at y position
            List<Rectangle> alignedBoundingBoxes = new List<Rectangle>(), tmpRes = new List<Rectangle>();
            for(var index = 0; index < boundingBoxes.Count;index++)
            {
                tmpRes.Clear();
                findAlignedBoxes(boundingBoxes, index, boundingBoxes.Count, tmpRes);
                if (tmpRes.Count > alignedBoundingBoxes.Count)
                {
                    alignedBoundingBoxes = tmpRes;
                }
            }

            //// sort bounding boxes from left to right
            alignedBoundingBoxes = alignedBoundingBoxes.OrderBy(c => c.X).ToList();

            if (_debugEdges)
            {
                // draw contours
                //Mat cont = M//zeros(edges.rows, edges.cols, CV_8UC1);
                //drawContours(cont, filteredContours, -1, Scalar(255));
                //imshow("contours", cont);
            }

            //// cut out found rectangles from edged image
            for (int i = 0; i < alignedBoundingBoxes.Count; ++i)
            {
                Rectangle roi = alignedBoundingBoxes[i];

                var img = this._imgGray.Clone();
                img.ROI = roi;
                img.Save(string.Format("J:\\output\\{0}_{1}.jpg", i,(new Random()).Next(100000)));
                _digits.Add(img);
                //if (_debugDigits)
                //{
                //    rectangle(_img, roi, Scalar(0, 255, 0), 2);
                //}
            }
        }

        private void findAlignedBoxes(List<Rectangle> list,int start,int end, List<Rectangle> temp)
        {
            var startRectangle = list[start];
            temp.Add(startRectangle);
            start = start + 1;
            for (var index= start; index< end; index++)
            {
                if (Math.Abs(startRectangle.Y - list[index].Y) < _config.digitYAlignment && Math.Abs(startRectangle.Height - list[index].Height) < 5)
                {
                    temp.Add(list[index]);
                }
            }
        }

        private float detectSkew()
        {
            throw new NotImplementedException();
        }

        private List<PointF> drawLines(List<PointF> lines)
        {
            throw new NotImplementedException();
        }

        private void drawLines(List<List<PointF>> lines, int xoff = 0, int yoff = 0)
        {
            throw new NotImplementedException();
        }

        private UMat cannyEdges()
        {            
            UMat edges = new UMat();
            Emgu.CV.CvInvoke.Canny(this._imgGray, edges, this._config.cannyThreshold1, this._config.cannyThreshold2);
            return edges;
        }

        private void filterContours(VectorOfVectorOfPoint contours, List<Rectangle> boundingBoxes, VectorOfVectorOfPoint filteredContours)
        {
            // filter contours by bounding rect size
            var count = contours.Size;
            for (int i = 0; i < count; i++)
            {
                Rectangle bounds = CvInvoke.BoundingRectangle(contours[i]);

                //if (bounds.Height > _config.digitMinHeight && bounds.Height < _config.digitMaxHeight
                //        && bounds.Width > 5 && bounds.Width < bounds.Height)
                //{
                if (true) { 
                    boundingBoxes.Add(bounds);
                    filteredContours.Push(contours[i]);
                }
            }
        }

        public void Dispose()
        {
        }
    }
}
