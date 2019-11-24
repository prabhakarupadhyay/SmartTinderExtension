var count = 0;
var clickButton = document.getElementById("clickButton");
var runInterval;
var swipeBool = false;

function triggerEvent() {
  var imgUrl = getImageUrl();
  var xhttp = new XMLHttpRequest();

  xhttp.open("POST", "http://127.0.0.1:5000/swipeData", true);

  xhttp.setRequestHeader("Content-Type", "application/json");

  xhttp.onreadystatechange = function() {
    if (this.readyState == 4 && this.status == 200) {
      console.log("Model Prediction:" + xhttp.responseText);
      if (JSON.parse(xhttp.responseText).predict == 1.0) {
        console.log("..................................................");
        console.log("Swiped Right");
        swipeRight();
      } else {
        console.log("..................................................");
        console.log("Swiped Left");
        swipeLeft();
      }
      console.log(JSON.parse(xhttp.responseText).predict);
    }
  };
  console.log("....................................................");
  console.log(
    "PLEASE WAIT WHILE THE IMAGE IS BEING VALIDATED BASED ON THE MODEL...."
  );
  xhttp.send(JSON.stringify({ Image_Url: imgUrl }));
}

chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
  if (request.message === "false") {
    document.addEventListener("keydown", checkKey);
  } else if (request.message === "true") {
    document.removeEventListener("keydown", checkKey);
  }

  if (request.message2 === "true") {
    trainDataServer();
  }
  if (request.message3 === "false") {
    swipeBool = true;
    triggerEvent();
  } else if (request.message3 === "true") {
    swipeBool = false;
    //clearInterval(runInterval);
  }
});

function checkKey(e) {
  e = e || window.event;

  if (e.keyCode == "38") {
    // up arrow
  } else if (e.keyCode == "40") {
    // down arrow
  } else if (e.keyCode == "37") {
    var imgUrl = getImageUrl();
    if (imgUrl != null) {
      sendDataServer(imgUrl, "Dislike");
    }
    // left arrow
    console.log("Left Click Count:" + count);
  } else if (e.keyCode == "39") {
    var imgUrl = getImageUrl();
    if (imgUrl != null) {
      sendDataServer(imgUrl, "Like");
    }
    // right arrow
    console.log("right Click Count:" + count);
  }
}

function getImageUrl() {
  var img = document.getElementsByClassName("StretchedBox")[4];
  if (img != undefined) {
    var style = img.currentStyle || window.getComputedStyle(img, false);
    var imgUrl = style.backgroundImage.slice(4, -1).replace(/"/g, "");
    if (imgUrl != "") {
      return imgUrl;
      //do work
    } else {
      console.log("image url blank skipping image");
    }
  } else {
    console.log("image undefined parsing scrollable image...");
    var img2 = document.getElementsByClassName(
      "profileCard__slider__img Z(-1)"
    )[0];
    if (img2 !== undefined) {
      var style2 = img2.currentStyle || window.getComputedStyle(img2, false);
      var imgUrl2 = style2.backgroundImage.slice(4, -1).replace(/"/g, "");
      if (imgUrl2 != "") {
        //do work
        return imgUrl2;
      } else {
        console.log("image url blank skipping image");
      }
    } else {
      console.log("skipping image");
    }
  }
  return null;
}

function sendDataServer(imgUrl, likeDislike) {
  var splitImg = imgUrl.split("/")[imgUrl.split("/").length - 1].split(".")[0];
  var newArr = [
    splitImg,
    "dummy",
    "20",
    "dummyLocation",
    "50km",
    "dummyJob",
    "dummySchool",
    likeDislike
  ];
  console.log(newArr);

  var xhttp = new XMLHttpRequest();

  xhttp.open("POST", "http://127.0.0.1:5000/browseData", true);

  xhttp.setRequestHeader("Content-Type", "application/json");

  xhttp.onreadystatechange = function() {
    if (this.readyState == 4 && this.status == 200) {
      console.log("Data stored successfully keep Swiping");
    }
  };
  console.log("....................................................");
  console.log("PLEASE WAIT WHILE THE DATA IS BEING STORED....");
  xhttp.send(JSON.stringify({ Image_Url: imgUrl, User_List: newArr }));
}

function trainDataServer() {
  console.log("Sending data to server.");
  var xhttp = new XMLHttpRequest();

  xhttp.open("POST", "http://127.0.0.1:5000/trainData", true);

  xhttp.setRequestHeader("Content-Type", "application/json");

  xhttp.onreadystatechange = function() {
    if (this.readyState == 4 && this.status == 200) {
      console.log(
        "Data trained successfully click Swipe icon from the panel to autoswipe profiles using ML."
      );
    }
  };
  console.log("....................................................");
  console.log("PLEASE WAIT WHILE THE DATA IS BEING TRAINED....");
  xhttp.send(JSON.stringify({ status: "train" }));
}

function swipeRight() {
  var curButton = $(document).find("button");
  $(curButton[4]).trigger("click");
  console.log("Swiped Right");
  if (swipeBool) {
    triggerEvent();
  } else {
    console.log("Stopped Auto Swipe");
  }
}

function swipeLeft() {
  var curButton = $(document).find("button");
  $(curButton[2]).trigger("click");
  console.log("Swiped Left");
  if (swipeBool) {
    triggerEvent();
  } else {
    console.log("Stopped Auto Swipe");
  }
}
