
#include "image_similarity_matching.h"

int main(void)
{
	vector<Mat> data = get_images();
	show_images("Original", data, 5);
	
	vector<vector<Mat> > histograms = get_histograms(data, 4, 5, 5);
	show_images("Histograms", histograms[1], 5);

	vector<Mat> color_match = get_top_three_color(histograms[0], data);
	show_images("COLOR", color_match, color_match.size()/8);


	while (1)
	{
		waitKey(100000);
	}
	return 0;
}

vector<Mat> get_top_three_color(vector<Mat> hist_list, vector<Mat> data)
{
	vector<Mat> color_match;
	for (size_t img1 = 0; img1 < data.size(); img1++)
	{
		cout << "Image1= i" << img1 + 1 << endl;
		color_match.push_back(data[img1]);
		int min_index[] = {40, 40, 40};
		for (size_t min_distance_no = 0; min_distance_no < 3; min_distance_no++)
		{

			float minimum = 10;

			cout << "Finding minimum " << min_distance_no + 1 << endl;
			for (size_t img2 = 0; img2 < data.size(); img2++)
			{
				if (img2 == img1 || img2 == min_index[0] || img2 == min_index[1] || img2 == min_index[2])
				{
					continue;
				}
				float l1_distance = get_L1_norm(hist_list[img1], hist_list[img2]);
				if (l1_distance < minimum)
				{
					minimum = l1_distance;
					min_index[min_distance_no] = img2;
				}
			}
			cout << "The no:" << min_distance_no + 1 << " minimum distance for image i" << img1 + 1 << " is " << minimum << " at index " << min_index[min_distance_no] << endl;
			color_match.push_back(data[min_index[min_distance_no]]);
		}
	}
	return color_match;
}

float get_L1_norm(Mat hist, Mat hist2)
{
	float sum = 0;
	for (size_t i = 0; i < hist.cols; i++)
	{
		sum += abs(hist.at<float>(i) - hist2.at<float>(i));
	}
	sum = sum / (2 * 60 * 89);
	return sum;
}
vector<vector<Mat> > get_histograms(vector<Mat> data, int blue_bins, int green_bins, int red_bins)
{
	vector<vector<Mat> > result;
	vector<Mat> hist_pictures;
	vector<Mat> hist_merge_list;
	for (size_t i = 0; i < data.size(); i++)
	{

		cout << "Getting Histograms for image" << (i + 1) << endl;
		Mat src = data[i];

		/// Separate the image in 3 places ( B, G and R )
		vector<Mat> bgr_planes;
		split(src, bgr_planes);

		/// Set the ranges ( for B,G,R) )
		float range[] = {0, 256};
		const float *histRange = {range};

		bool uniform = true;
		bool accumulate = false;

		Mat hist;

		// convert to grayscale
		Mat image_gray;
		cvtColor(src, image_gray, COLOR_BGR2GRAY);

		// get image threshold
		Mat image_mask;
		threshold(image_gray, image_mask, 127, 255, THRESH_BINARY);

		int channels[] = {0, 1, 2};
		int histSize[] = {blue_bins, green_bins, red_bins};
		float b_ranges[] = {0, 256};
		float g_ranges[] = {0, 256};
		float r_ranges[] = {0, 256};
		const float *ranges[] = {b_ranges, g_ranges, r_ranges};
		/// Compute the  combined histograms:
		calcHist(&src, 1, channels, image_mask, hist, 3, histSize, ranges, true, false);

		/// Set the ranges ( for B,G,R) )
		float range_BGR[] = {0, 256};
		const float *hist_Range_BGR = {range};
		Mat b_hist, g_hist, r_hist;
		// Compute the RGB histograms for drawing
		calcHist(&bgr_planes[0], 1, 0, image_mask, b_hist, 1, &blue_bins, &hist_Range_BGR, uniform, accumulate);
		calcHist(&bgr_planes[1], 1, 0, image_mask, g_hist, 1, &green_bins, &hist_Range_BGR, uniform, accumulate);
		calcHist(&bgr_planes[2], 1, 0, image_mask, r_hist, 1, &red_bins, &hist_Range_BGR, uniform, accumulate);

		// Draw the histograms for B, G and R
		int hist_w = 512;
		int hist_h = 400;
		int bin_w_b = cvRound((double)hist_w / blue_bins);
		int bin_w_g = cvRound((double)hist_w / green_bins);
		int bin_w_r = cvRound((double)hist_w / red_bins);

		Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

		/// Normalize the result to [ 0, histImage.rows ]
		normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
		normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
		normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

		Mat hist_1d = hist.reshape(1, 1);

		/// Draw for each channel
		for (int i = 1; i < blue_bins; i++)
		{
			line(histImage, Point(bin_w_b * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
				 Point(bin_w_b * (i), hist_h - cvRound(b_hist.at<float>(i))),
				 Scalar(255, 0, 0), 2, 8, 0);
		}

		for (int i = 1; i < green_bins; i++)
		{
			line(histImage, Point(bin_w_g * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
				 Point(bin_w_g * (i), hist_h - cvRound(g_hist.at<float>(i))),
				 Scalar(0, 255, 0), 2, 8, 0);
		}

		for (int i = 1; i < red_bins; i++)
		{
			line(histImage, Point(bin_w_r * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
				 Point(bin_w_r * (i), hist_h - cvRound(r_hist.at<float>(i))),
				 Scalar(0, 0, 255), 2, 8, 0);
		}

		hist_merge_list.push_back(hist_1d);
		hist_pictures.push_back(histImage);
	}

	result.push_back(hist_merge_list);
	result.push_back(hist_pictures);
	cout << "RETURNING\n";
	return result;
}

void show_images(const char *title, vector<Mat> data, int rows)
{
	const int WINDOW_SIZE = 700;
	Mat result = makeCanvas(data, WINDOW_SIZE, rows);
	namedWindow(title, WINDOW_NORMAL);
	imshow(title, result);
}

vector<Mat> get_images()
{
	cout << "Loading images\n";

	// get image path
	String path("images/i*.ppm");
	vector<String> file_name;
	vector<Mat> data;

	// get all file names in path
	glob(path, file_name, true);

	for (size_t i = 0; i < file_name.size(); i++)
	{
		cout << "Read in " << file_name[i] << endl;
		Mat image = imread(file_name[i]);
		if (image.empty())
		{
			die("Image is empty\n");
		}
		data.push_back(image);
	}

	return data;
}

/*
void show_data(const char *title, vector<Mat> &data)
{
	Mat image = data[0].clone();
	namedWindow(title, WINDOW_NORMAL);
	for (size_t i = 1; i < data.size(); i++)
	{
		
		hconcat(data[i], image);
	}
	imshow(title, image);
	
	waitKey(0);




	

		
		


}*/
