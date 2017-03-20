
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

class SimpleGraph
{
public:
	SimpleGraph(int width, int height, int count, int padding = 50) : axis_colour(55, 55, 55), pen(255, 255, 255), eraser(0, 0, 0)
	{
		this->width = width;
		this->height = height;
		this->max_data_points = count;
		this->padding = padding;

		this->font_face = cv::FONT_HERSHEY_COMPLEX;
		this->font_scale = 0.5;
		this->baseline = 0;

		this->intermediate_labels_count = 5;
		this->InitialiseCanvas();
	}

	void InitialiseCanvas()
	{
		this->canvas = cv::Mat(cv::Size(this->width, this->height), CV_8UC3);
		this->DrawAxis();
	}

	void DrawAxis()
	{
		this->x_start = cv::Point(0 + this->padding, this->height / 2);
		this->y_start = cv::Point(0 + this->padding, 0 + this->padding);
		this->x_end = cv::Point(this->width - this->padding, this->height / 2);
		this->y_end = cv::Point(0 + this->padding, this->height - this->padding);

		this->DrawLine(this->x_start, this->x_end, this->axis_colour);
		this->DrawLine(this->y_start, this->y_end, this->axis_colour);
	}

	void DrawLine(cv::Point p1, cv::Point p2, cv::Scalar& colour, int thickness = 1, int line_type = 8, int shift = 0)
	{
		cv::line(this->canvas, p1, p2, colour, thickness, line_type, shift);
	}

	void DrawLines(std::vector<cv::Point> p, cv::Scalar& colour, int thickness = 1, int line_type = 8, int shift = 0)
	{
		for (int i = 0; i < p.size() - 1; ++i) {
			cv::line(this->canvas, p.at(i), p.at(i + 1), colour, thickness, line_type, shift);
		}
	}

	template<typename T>
	T PercentageDifference(T a, T b) {
		return abs(((a - b) / ((a + b) / 2)));
	}

	void DrawPeakValley() {
		float peak = 0, valley = 1;
		int peak_pos = 0, valley_pos = 0;

		for (int i = 0; i < this->normalised_points.size() - 1; ++i) {
			float value = this->normalised_points.at(i), next = this->normalised_points.at(i + 1);
			if (value > next && value > peak && this->PercentageDifference(value, peak) > 0.10) {
				peak = value;
				peak_pos = i;
			}

			if (value < next && value < valley && this->PercentageDifference(value, valley) > 0.10) {
				valley = value;
				valley_pos = i;
			}
		}

		int x_gap = (this->x_end.x - this->x_start.x) / this->max_data_points;
		int y_gap = (this->y_end.y - this->y_start.y);

		int p_pos_x = this->x_start.x + (peak_pos * x_gap) + 1;
		int v_pos_x = this->x_start.x + (valley_pos * x_gap) + 1;

		this->DrawLine(cv::Point(v_pos_x, this->padding), cv::Point(v_pos_x, this->height / 2), cv::Scalar(0, 0, 255));
		this->DrawLine(cv::Point(p_pos_x, this->height / 2), cv::Point(p_pos_x, this->height - this->padding), cv::Scalar(0, 255, 255));
	}

	template<typename T>
	std::string LabelToString(T in) {
		std::stringstream stream;
		stream << std::fixed << std::setprecision(2) << in;
		return stream.str();
	}

	void DrawText(std::string label, cv::Point p) {
		cv::putText(this->canvas, label, p, this->font_face, this->font_scale, this->pen);
	}

	void DrawText(std::string label, cv::Point p, cv::Scalar colour) {
		cv::putText(this->canvas, label, p, this->font_face, this->font_scale, colour);
	}

	void DrawLabels() {
		std::string y_min_lab = this->LabelToString(this->min_y);
		std::string y_max_lab = this->LabelToString(this->max_y);
		std::string x_min_lab = this->LabelToString(this->min_x);;
		std::string x_max_lab = this->LabelToString(this->max_data_points);

		cv::Size y_min_text_size = this->GetTextSize(y_min_lab);
		cv::Size y_max_text_size = this->GetTextSize(y_max_lab);
		cv::Size x_min_text_size = this->GetTextSize(x_min_lab);
		cv::Size x_max_text_size = this->GetTextSize(x_max_lab);

		int y_min_text_err = this->GetTextError(y_min_text_size.width);
		int y_max_text_err = this->GetTextError(y_max_text_size.width);
		int x_min_text_err = this->GetTextError(x_min_text_size.width);
		int x_max_text_err = this->GetTextError(x_max_text_size.width);

		cv::Point y_min_pos(this->y_end.x - (y_min_text_size.width / 2) + y_min_text_err, this->y_end.y + (y_min_text_size.height / 2));
		cv::Point y_max_pos(this->y_start.x - (y_max_text_size.width / 2) + y_max_text_err, this->y_start.y + (y_max_text_size.height / 2));
		cv::Point x_min_pos(this->x_start.x + x_min_text_err, this->x_start.y + (x_min_text_size.height / 2));
		cv::Point x_max_pos(this->x_end.x + x_max_text_err, this->x_end.y + (x_max_text_size.height / 2));

		this->DrawText(y_min_lab, y_min_pos);
		this->DrawText(y_max_lab, y_max_pos);
		this->DrawText(x_min_lab, x_min_pos);
		this->DrawText(x_max_lab, x_max_pos);

		//Draw Intermitant Points
		int x_diff = (x_max_pos.x - x_min_pos.x) / this->intermediate_labels_count;
		int y_diff = (y_max_pos.y - y_min_pos.y) / this->intermediate_labels_count;

		int x_val_diff = (this->max_x - this->min_x) / this->intermediate_labels_count;
		int y_val_diff = (this->max_y - this->min_y) / this->intermediate_labels_count;

		for (int i = 1; i < this->intermediate_labels_count; i++) {
			int x_step = x_diff * i, y_step = y_diff * i;
			int x_val_step = x_val_diff * i, y_val_step = y_val_diff * i;

			std::string x_label = this->LabelToString(this->min_x + x_val_step);
			std::string y_label = this->LabelToString(this->min_y + y_val_step);

			cv::Point x_p(x_min_pos.x + x_step, x_min_pos.y);
			cv::Point y_p(y_min_pos.x, y_min_pos.y + y_step);

			this->DrawText(x_label, x_p, this->axis_colour);
			this->DrawText(y_label, y_p, this->axis_colour);
		}
	}

	int GetTextError(int width) {
		return this->padding - width <= 0 ? abs(width - this->padding) : 0;
	}

	cv::Size GetTextSize(std::string label) {
		return cv::getTextSize(label, this->font_face, this->font_scale, 1, 0);
	}

	void AddData(float datapoint)
	{
		this->data_points.push_back(datapoint);

		if (this->data_points.size() > this->max_data_points) {
			this->data_points.erase(this->data_points.begin(), this->data_points.begin() + this->data_points.size() - this->max_data_points);
		}

		this->GetMinMax();
		this->NormaliseData();
		this->ConvertNormalisedToPoints();
		this->Invalidate();
	}

	void Invalidate() {
		this->canvas.setTo(0);
		this->DrawAxis();
		this->DrawLabels();
		this->DrawLines(this->points, this->pen);
		this->DrawPeakValley();
	}

	void GetMinMax() {
		this->min_x = 0;
		this->max_x = this->max_data_points;
		this->min_y = *std::min_element(std::begin(this->data_points), std::end(this->data_points));
		this->max_y = *std::max_element(std::begin(this->data_points), std::end(this->data_points));
	}

	void NormaliseData() {
		this->normalised_points = std::vector<float>(this->data_points.size());

		for (int i = 0; i < this->data_points.size(); ++i) {
			this->normalised_points.at(i) = (this->data_points.at(i) - this->min_y) / (this->max_y - this->min_y);
		}
	}

	void ConvertNormalisedToPoints() {
		int x_gap = (this->x_end.x - this->x_start.x) / this->max_data_points;
		int y_gap = (this->y_end.y - this->y_start.y);

		this->points = std::vector<cv::Point>(this->normalised_points.size());

		for (int i = 0; i < normalised_points.size(); ++i) {
			int x = 0, y = 0;
			x = this->x_start.x + (i * x_gap) + 1;
			y = this->y_start.y + normalised_points.at(i) * y_gap;

			this->points.at(i) = cv::Point(x, y);
		}
	}

	void Show()
	{
		cv::imshow("SimpleGraph", canvas);
	}
private:
	cv::Mat canvas;
	std::vector<float> data_points, normalised_points;
	std::vector<cv::Point> points;
	int max_data_points, width, height, padding, font_face, intermediate_labels_count;
	float min_x, min_y, max_x, max_y, font_scale, baseline;
	cv::Point y_start, x_start, y_end, x_end;
	cv::Scalar axis_colour, pen, eraser;
};
