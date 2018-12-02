<!DOCTYPE html>
<html>
<head>
  <title>Plant Seedlings Classification</title>
</head>
<body>
<table border="0">

<h1> Plant seedlings classification</h1>
  
<p> Upload your pictures with seedlings and you will find out the species.</p>
<img src="seedlings.jpg" height="500"/></br></br></br>
  <form enctype="multipart/form-data" action="abc.php" method="POST">
    <p>Upload your file</p>
    <input type="file" name="uploaded_file"></input><br />
    <input type="submit" value="Upload"></input>
  </form>
</body>
</html>
<?php
  if(!empty($_FILES['uploaded_file']))
  {
    $path = "test/";
    $path = $path . basename( $_FILES['uploaded_file']['name']);
	
	$ext_error= false;
	$extentions= array('jpg','jpeg','gif','png');
	$file_ext=explode('.', $_FILES['uploaded_file']['name']);
	$file_ext=end($file_ext);

	if (!in_array($file_ext, $extentions)){
		$ext_error=true;
	}

	if ($ext_error){
		echo "Invalid file!";
		exit(0);
	}
	else {
		echo "Successful uploading!";
	}

    if(move_uploaded_file($_FILES['uploaded_file']['tmp_name'], $path)) {
      echo "The file ".  basename( $_FILES['uploaded_file']['name']). 
      " has been uploaded";
      echo "<br>";
    } else{
        echo "There was an error uploading the file, please try again!";
        echo "<br>";
    }
    ini_set('max_execution_time', 0);
    $command = escapeshellcmd('python plantseed_CNN.py');
	$output = shell_exec($command);
	$rest = substr($output, strpos($output, "Prediction"));
	echo '<h3>';
	echo $rest;
	echo '</h3>';
	$history = "history/";
	$history = $history . basename( $_FILES['uploaded_file']['name']);
	copy($path, $history);
	echo "<br>";
	echo '<img src= "' .'history/' .$_FILES['uploaded_file']['name'] . '"/>';
	unlink('test/'.$_FILES['uploaded_file']['name']);
  }
?>

