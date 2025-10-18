const int ledPin= 8;

void setup() {
  Serial.begin(9600);
  pinMode(ledPin, OUTPUT);
}

void loop() {
  if (Serial.available() > 0){
    String msg = Serial.readString();

    if (msg == "ON"){
      digitalWrite(ledPin, HIGH);
    }

    else if(msg == "OFF"){
      digitalWrite(ledPin, LOW);
    }
  }
}
