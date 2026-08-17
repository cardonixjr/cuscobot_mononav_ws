std_msgs/Header header

float32 frame_time
float32 preprocessing_time
float32 detection_time
float32 matching_time
float32 pose_estimation_time

float32 fps

uint32 detected_keypoints
uint32 raw_matches
uint32 good_matches
uint32 inliners

float32 match_ratio
float32 inliner_ratio

float32 cpu_usage
float32 memory_usage
float32 cpu_temperature

bool tracking_ok