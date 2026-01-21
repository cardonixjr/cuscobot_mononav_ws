#include <math.h>

// ROS libs
#include <ros.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Empty.h>
#include <std_msgs/Int32.h>
#include <std_msgs/Float32.h>
#include <geometry_msgs/Twist.h>

// MD49 commands
#define CMD (byte)0x00                             
#define GET_SPEED1 0x21
#define GET_ENC1 0x23
#define GET_ENC2 0X24
#define SET_SPEED1 0x31
#define SET_SPEED2 0x32
#define ENC_RESET 0x35
#define DISABLE_TIMEOUT  0X38


int ledPin = 13;
bool ledState = HIGH;


// Robot dimensions
float WHEEL_RADIUS = 0.06;
float BASE_LENGTH = 0.37;
int MAX_PWM = 40;

// Encoder
uint32_t encoder = 0;
byte enc1a, enc1b, enc1c, enc1d = 0;

/********** DEFINIÇÃO DE TOPICOS ROS **********/
ros::NodeHandle  nh;

// Publishers
// left encoder publisher
std_msgs::Int32 leftEncoder;
ros::Publisher leftEncoderPublisher("left_encoder_pulses", &leftEncoder);

// right encoder publisher
std_msgs::Int32 rightEncoder;
ros::Publisher rightEncoderPublisher("right_encoder_pulses", &rightEncoder);

void cmdVelCallback(const geometry_msgs::Twist& cmd_vel){
  /* MD49 specifications
    RPM: from -116.5 to 116.6
    PWM: from 0 to 255  (the zero speed is the PWM 128)
    m/s: from -0.7326 to 0.7326 
  */
  // PWM calc variables
  int left_pwm;
  int right_pwm;
  float linear;
  float angular;

  linear = cmd_vel.linear.x;
  angular = cmd_vel.angular.z;

  float right_linear = ((linear*2) + (angular*BASE_LENGTH))/2;
  float left_linear =  (linear*2) - right_linear;

  left_pwm = (int) round((left_linear + 0.7326) * (255 - 0) / (0.7326 + 0.7326));
  right_pwm = (int) round((right_linear + 0.7326) * (255 - 0) / (0.7326 + 0.7326));

  Serial1.write(CMD);
  Serial1.write(SET_SPEED1);
  Serial1.write(left_pwm);
    
  Serial1.write(CMD);
  Serial1.write(SET_SPEED2);
  Serial1.write(right_pwm);

  // ledState = !ledState;
  // digitalWrite(ledPin, ledState);
}

void resetEncoderCB(const std_msgs::Empty &command){
  Serial1.write(CMD);
  Serial1.write(ENC_RESET);
}

ros::Subscriber<geometry_msgs::Twist> cmdVelSubscriber("cmd_vel", cmdVelCallback );
ros::Subscriber<std_msgs::Empty> encoderResetSubscriber("reset_encoder", resetEncoderCB);


void setup(){ 
  // Serial
  Serial1.begin(9600);
  Serial1.write(CMD);
  Serial1.write(ENC_RESET);
  // Serial.begin(9600);

  pinMode(ledPin, OUTPUT);

  // ROS
  nh.initNode();
  nh.advertise(leftEncoderPublisher);
  nh.advertise(rightEncoderPublisher);
  nh.subscribe(cmdVelSubscriber);
  nh.subscribe(encoderResetSubscriber);

  delay(500);
}


void loop(){ 
  nh.spinOnce();

  /********** Encoder Reading ***********/
  Serial1.write(CMD);
  Serial1.write(GET_ENC1); // Recieve encoder 1 value
  // delay(50);
  while(Serial1.available()<=3);
  if (Serial1.available())
  {
    enc1a = Serial1.read();
    enc1b = Serial1.read();
    enc1c = Serial1.read();
    enc1d = Serial1.read();
  }
  encoder = (((uint32_t)enc1a << 24) +
  ((uint32_t)enc1b << 16) +
  ((uint32_t)enc1c << 8) +
  ((uint32_t)enc1d << 0));
  leftEncoder.data = (uint32_t) encoder;
  
  // Read Right Encoder 
  Serial1.write(CMD);
  Serial1.write(GET_ENC2); // Recieve encoder right value
  // delay(50);
  while(Serial1.available()<=3);
  if (Serial1.available() > 3)
  {
    enc1a = Serial1.read();
    enc1b = Serial1.read();
    enc1c = Serial1.read();
    enc1d = Serial1.read();
  }
  encoder = (((uint32_t)enc1a << 24) +
  ((uint32_t)enc1b << 16) +
  ((uint32_t)enc1c << 8) +
  ((uint32_t)enc1d << 0));
  rightEncoder.data = (uint32_t) encoder;
  nh.spinOnce();

    /********** Publish data ***********/
  leftEncoderPublisher.publish(&leftEncoder);
  rightEncoderPublisher.publish(&rightEncoder);
}

