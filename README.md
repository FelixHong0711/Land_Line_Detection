# Lane Line Detection Project

As we navigate the road, our eyes serve as the key navigational tools, interpreting the environment around us. The lane lines on the road act as a dependable guide, offering crucial cues on steering direction. In the realm of self-driving car development, a pivotal objective is leveraging sophisticated algorithms to autonomously recognize these lane lines, enhancing the vehicle's ability to navigate with precision and safety. This project is designed to address these concerns by automatically detecting lane lines and identifying eligible driving areas.

Hereby is the reference link for the project outcome: http://ec2-54-163-60-152.compute-1.amazonaws.com:5000/

# Pipeline
In general, the model for detection will contain 6 main steps:
1. Identify regions of the image that correspond to the colors white and yellow, often seen in road markings.
2. Focus on the relevant part of the image, usually the road area where the lane markings are expected - region of interest.
3. Convert image to grayscale
4. Reduce noise of image by blurring it by a Gaussian function
5. Detect a wide range of edges in image using the Canny edge detector
6. Detect straight lines using Hough Line Transform