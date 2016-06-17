#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;
/*
 * =====================================================================================
 *        Class:  ColorSketch
 *  Description:  
 * =====================================================================================
 */
class ColorSketch
{
	public:
		ColorSketch();                     /* constructor */
		ColorSketch(int s32AmtDark);                             /* constructor */
		void Generate(Mat &sMatInput, Mat &sMatOutput);
		~ColorSketch(){};

	private:
		class Params{
			public:
				Params();
				const int s32KernelSize;
				const int s32NumDirs;
				int s32AmtDark;
		};
		Params sSketchParams;

		Mat sMatInputGray;
		Mat sMatLineSketch;// Store the Line Sketch
		Mat sMatToneMap; // Store the Tone Map
		Mat sMatFinalSketch;

		void init(int s32AmtDark);
		void GenerateLineDrawing(const Mat &psMatInputGray, Mat &psMatLineSketch);
		void GenerateToneMap(const Mat &psMatInputGray, Mat &psMatToneMap);
		void GenerateSketchCurve(vector<double> &vecHist);
		void GenerateFinalSketch(const Mat &psMatLineSketch, const Mat &psMatToneMap, Mat &psMatFinalSketch);
		//void ShowHistogram(Mat &psMatHist);

}; /* -----  end of class ColorSketch  ----- */
