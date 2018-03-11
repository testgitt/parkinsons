<?php
echo "<a href=http://localhost/woman/new/home.php>GO TO TEST PAGE</a>";
echo "<br></br><br></br>";
$output = shell_exec("/usr/bin/python m2.py");
echo "<pre>$output</pre>";  
 
echo '<img src="hist.png" border=0>';
 echo '<img src="algcomp.png" border=0>';
 echo '<img src="lr.png" border=0>';
 echo '<img src="tree.dot" border=0>';




?>
