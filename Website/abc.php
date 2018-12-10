<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Plant Seedlings Classification</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>


 <h1> Plant Seedlings Classification</h1>


    <ul class="menu">
      <li><a class="ajax-link" href="abc.php">Home</a></li>
      <li><a class="ajax-link" href="https://www.kaggle.com/c/plant-seedlings-classification">Projects</a></li>
      <li><a class="ajax-link" href="contact.html">Contact</a></li>
    </ul>

</div>



<div class="text-content">
<p> Upload your pictures of plant seedlings and you will find out the species.</p>
</div>

<img src="seedlings.jpg" align="middle" height="400" width="600" /></br></br>

  <form enctype="multipart/form-data" action="abc.php" method="POST">
    <p>Upload your file</p>
    <div align="center">

     <a href="javascript:;" class="file">choose file
    <input type="file" name="uploaded_file">
     </a>

     <input type="submit" value="Upload"></input><br/>
    </div>
  </form>


</body>

<?php

  if(!empty($_FILES['uploaded_file']))
  {
    $path = "/Applications/XAMPP/xamppfiles/htdocs/PlantSeedlings/test/";
    $path = $path . basename( $_FILES['uploaded_file']['name']);
	
	$ext_error= false;
	$extentions= array('jpg','jpeg','gif','png');
	$file_ext=explode('.', $_FILES['uploaded_file']['name']);
	$file_ext=end($file_ext);

	if (!in_array($file_ext, $extentions)){
		$ext_error=true;
	}

	if ($ext_error){
		echo "<p align='center'>Invalid file!</p>";
		exit(0);
	}
	else {
		echo "<p align='center'> Successfully uploaded!!</p>";
	}
	//move_uploaded_file($_FILES['uploaded_file']['tmp_name'], $path);
    if(move_uploaded_file($_FILES['uploaded_file']['tmp_name'], $path)) {
      echo "<p align='center'>The file ".  basename( $_FILES['uploaded_file']['name']). 
      " has been uploaded</p>";
      echo "<br>";
    } else {
        echo "<p align='center'>There was an error uploading the file, please try again!</p>";
        echo "<br>";
    }
    ini_set('max_execution_time', 100);
    $command = escapeshellcmd('python plantseed_CNN.py');
	$output = shell_exec($command);
	$rest = substr($output, strpos($output, "Prediction"));
	echo '<h3>';
	echo "<p align='center'>$rest</p>";
	echo '</h3>';
	$history = "history/";
	$history = $history . basename( $_FILES['uploaded_file']['name']);
	copy($path, $history);
	echo "<br>";
	echo '<img src= "' .'history/' .$_FILES['uploaded_file']['name'] . '"/>';
	unlink('test/'.$_FILES['uploaded_file']['name']);
   }
?>

<footer>
  <div class="footer-margin">
    <div class="copyright">© Copyright 2018 Yang Qiao, Haotian Cheng, Xueyao Liang. All Rights Reserved.</div>
</footer>

</html>

