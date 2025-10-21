int buttonPin = 5;
int irPin = 4;
 
int buttonState;
int irState;
 
void setup() {
  Serial.begin(9600);  // Start serial communication at 9600 baud
  pinMode(buttonPin, INPUT);
  pinMode(irPin, INPUT);
}
 
void loop() {
  buttonState = digitalRead(buttonPin);
  irState = digitalRead(irPin);
 
  if (buttonState == HIGH) {
    Serial.println("ESTOP");
  }
 
  if (irState == LOW) {
    Serial.println("LIGHT CURTAIN");
  }
}