
// Inputs
int a_in = 7;
int b_in = 6;
int c_in = 5;
int d_in = 4;

// Outputs
int a_out = 12;
int b_out = 11;
int c_out = 10;
int d_out = 9;

// Status handlers
int status_a = 0;
int status_b = 0;
int status_c = 0;
int status_d = 0;

void setup() {
  Serial.begin(9600);
  Serial.println("ALTECHNOLOGIES Firmware Version 4.0");
  Serial.println("COMMANDS: [CAM1, CAM2, CAM3, CAM4, LEDTEST]");
  // PIN Setup
  pinMode(a_in, OUTPUT);
  pinMode(b_in, OUTPUT);
  pinMode(c_in, OUTPUT);
  pinMode(d_in, OUTPUT);

  pinMode(a_out, OUTPUT);
  pinMode(b_out, OUTPUT);
  pinMode(c_out, OUTPUT);
  pinMode(d_out, OUTPUT);
  
}

void loop() {
  while(Serial.available()>0){
    String cmd = Serial.readString();

    if(cmd=="LEDTEST\n"){
        digitalWrite(a_out, HIGH);
        delay(100);
        digitalWrite(b_out, HIGH);
        delay(100);
        digitalWrite(c_out, HIGH);
        delay(100);
        digitalWrite(d_out, HIGH);
        delay(100);
        digitalWrite(a_out, LOW);
        delay(100);
        digitalWrite(b_out, LOW);
        delay(100);
        digitalWrite(c_out, LOW);
        delay(100);
        digitalWrite(d_out, LOW);
        delay(200);
        digitalWrite(a_out, HIGH);
        digitalWrite(b_out, HIGH);
        digitalWrite(c_out, HIGH);
        digitalWrite(d_out, HIGH);
        delay(200);
        digitalWrite(a_out, LOW);
        digitalWrite(b_out, LOW);
        digitalWrite(c_out, LOW);
        digitalWrite(d_out, LOW);
    }
    
    if(cmd=="CAM1\n"){
      status_a = 1;
    }

    if(cmd=="CAM2\n"){
      status_b = 1;
    }

    if(cmd=="CAM3\n"){
      status_c = 1;
    }

    if(cmd=="CAM4\n"){
      status_d = 1;
    }
  }

  if(digitalRead(a_in)){
    status_a = 0;
  }

  if(digitalRead(b_in)){
    status_b = 0;
  }

  if(digitalRead(c_in)){
    status_c = 0;
  }

  if(digitalRead(d_in)){
    status_d = 0;
  }

  if(status_a){
      digitalWrite(a_out, !digitalRead(a_out));
  }else{
      digitalWrite(a_out, LOW);
  }

  if(status_b){
      digitalWrite(b_out, !digitalRead(b_out));
  }else{
      digitalWrite(b_out, LOW);
  }

  if(status_c){
      digitalWrite(c_out, !digitalRead(c_out));
  }else{
      digitalWrite(c_out, LOW);
  }

  if(status_d){
      digitalWrite(d_out, !digitalRead(d_out));
  }else{
      digitalWrite(d_out, LOW);
  }

  delay(40);
}
