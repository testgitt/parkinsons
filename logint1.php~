<!DOCTYPE HTML>

<html>
	
	<body>

	

	


<style>
.error {color: #FF0000;}
</style>
<body> 
<!DOCTYPE HTML>
<html>
<head>
<style>
.error {color: #FF0000;}
</style>
</head>
<body> 

<?php
// define variables and set to empty values
//$USNErr = $BUSROUTEErr = $PICKUPPOINTErr = $NAMEErr = "";
$nameErr = $addressErr  = $emailErr = $learnErr= $passwordErr=$cpasswordErr= $phoneErr= $domainErr="";
//$password;
//$phone=0;
//$name = $address  = $phoneErr= $passwordErr= $email = $learn="";
//$flag=0;

if ($_SERVER["REQUEST_METHOD"] == "POST") {
  /* if (empty($_POST["name"])) {
     $nameErr = "name is required";
   } else {
     $name = test_input($_POST["name"]);$name = test_input($_POST["name"]);
if (!preg_match("/^[a-zA-Z ]*$/",$name)) {
  $nameErr = "Only letters and white space allowed"; 
}
	 
   }
   
   if (empty($_POST["address"])) {
     $addressErr = "address is required"; 
   } else {
	 $address = test_input($_POST["address"]);
	 
   } */
     
   if (empty($_POST["email"])) {
     $emailErr = "email is required";
   } else {
     $email = test_input($_POST["email"]);
	 $email = test_input($_POST["email"]);
if (!filter_var($email, FILTER_VALIDATE_EMAIL)) {
  $emailErr = "Invalid email format"; 
}
   } 
   //if (empty($_POST["phone"])) {
   //  $phoneErr = "phone no is required";
   //} else {
   //  $phone = test_input($_POST["phone"]);
   //}
    if (empty($_POST["password"])) {
     $passwordErr = "password is required";
   } else {
     $password = test_input($_POST["password"]);
   }
   if (empty($_POST["cpassword"])) {
     $passwordErr = "password is required";
   } else {
     $password = test_input($_POST["password"]);
   }
  /* if (empty($_POST["domain"])){
    $domainErr = "domain is required";
  } else {
    $domain = test_input($_POST["domain"]);
   }*/
   
    
   

   
}

function test_input($data) {
	$data = trim($data);
   $data = stripslashes($data);
	
   
   
   $data = htmlspecialchars($data);
   return $data;
}
?>


<p><span class="error">* required field.</span></p>
<form method="post" action="<?php echo htmlspecialchars($_SERVER["PHP_SELF"]);?>"> 
 
  
   EMAIL-ID: <input type="email" name="email">
   <span class="error">* <?php echo $emailErr;?></span>
   <br><br> 
   PASSWORD: <input type="password" name="password">
   <span class="error">* <?php echo $passwordErr;?></span>
   <br><br> 
   CONFIRM PASSWORD: <input type="password" name="cpassword">
   <span class="error">* <?php echo $cpasswordErr;?></span>
   <br><br> 
  
   
  
   <input type="submit" name="submit" value="Submit"> 
   
</form>

<?php

//$pass=$_POST['password'];
//$enpass=md5($pass);
?>
<?php
if (isset($_POST['submit'])) {
	
	if($_POST['cpassword']==$_POST['password'] && $nameErr == "" && $emailErr=="" && $passwordErr=="" && $cpasswordErr=="" ){
	
	$pass=$_POST['password'];
$enpass=md5($pass);
$servername = "localhost";
$username = "root";
$password = "";
$dbname = "woman";

// Create connection
$conn = new mysqli($servername, $username, $password, $dbname);
// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
} 


//$sql = "INSERT INTO learner(name,address,phone,email,password,learn,domain)
//VALUES ('$_POST[name]', '$_POST[address]','$_POST[phone]' ,'$_POST[email]', '$enpass','$_POST[learn]','$_POST[domain]')";
$sql = "SELECT * from teacher where email='$_POST[email]' and password='$enpass'";
$result = $conn->query($sql);

if ($result->num_rows > 0) {
		echo "successful login";
		echo "<br></br>";
   // echo "<table><tr><th>ID</th><th>Name</th></tr>";
    // output data of each row
   /*while($row = $result->fetch_assoc()) {
       echo "<tr><td>".$row["domain"]."</td><td>".$row["learn"]." ".$row["name"]."</td></tr>";
	  $dom=$row['domain'];
	   echo $dom;
	   $ler=$row['learn'];*/
	   
	   //echo $_POST[email];
	   
	   $sql1 = "SELECT * from review where teacher_email='$_POST[email]' ";
$result1 = $conn->query($sql1);

if ($result1->num_rows > 0) {
		echo "YOUR REVIEWS";
		 echo "<br></br>";
		
   //echo "<table><tr><th>ID</th><th>Name</th></tr>";
    // output data of each row
    while($row1 = $result1->fetch_assoc()) {
       //echo "<tr><td>".$row1["domain"]."</td><td>".$row1["teach"]." ".$row1["name"]."</td></tr>";
	  echo "RATINGS :".$row1["rating"]."<br></br> REVIEW/COMMENTS:".$row1["comments"]."<br></br>" ;
	   
	   
	//echo "successful login";
	
    }
}

   // echo "</table>";
 else {
    echo "0 results";
}
	   
	  
	//echo "successful login";
	
    

   // echo "</table>";
   
	
 

//if ($conn->query($sql) === TRUE) {
 //   echo "New record created successfully";
//} else {
//    echo "Error: " . $sql . "<br>" . $conn->error;
//}
 
   } 

$conn->close();
}
}
else{ echo "confirm password not same";}
?>
</body>
</html>
</body>
</html>
