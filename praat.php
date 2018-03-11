<html>
<body>
<!--<h4>Select the sound file you just downloaded and run the popped up script and save the output on Info window as csv file</h4>-->
<?php
//exec ("./praat");

shell_exec("xterm -hold -e './praat.sh'");



?>
<a href="http://localhost/woman/new/home.php">TEST YOUR VOICE RECORDING</a>
</html>
</body>
