var browseButton = document.getElementById("browseButton");
var trainButton = document.getElementById("trainButton");
var swipeButton = document.getElementById("swipeButton");
var Start = document.getElementById("Start");
var train = document.getElementById("train");
var swipe = document.getElementById("swipe");
var intervalRun = false;
var intervalRun2 = false;

browseButton.onclick = function(e) {
  if (intervalRun) {
    chrome.tabs.query({ currentWindow: true, active: true }, function(tabs) {
      var activeTab = tabs[0];
      chrome.tabs.sendMessage(activeTab.id, { message: "true" });
    });
    intervalRun = false;
    Start.innerHTML = "Browse Again";
    Start.style.left = "23px";
  } else {
    Start.innerHTML = "Running..";

    chrome.tabs.query({ currentWindow: true, active: true }, function(tabs) {
      var activeTab = tabs[0];
      chrome.tabs.sendMessage(activeTab.id, { message: "false" });
    });
    intervalRun = true;
  }
};

trainButton.onclick = function(e) {
  train.innerHTML = "Running...";

  chrome.tabs.query({ currentWindow: true, active: true }, function(tabs) {
    var activeTab = tabs[0];
    chrome.tabs.sendMessage(activeTab.id, { message2: "true" });
  });
};

swipeButton.onclick = function(e) {
  if (intervalRun2) {
    chrome.tabs.query({ currentWindow: true, active: true }, function(tabs) {
      var activeTab = tabs[0];
      chrome.tabs.sendMessage(activeTab.id, { message3: "true" });
    });
    intervalRun2 = false;
    swipe.innerHTML = "Swipe Again";
    swipe.style.left = "23px";
  } else {
    swipe.innerHTML = "Running...";

    chrome.tabs.query({ currentWindow: true, active: true }, function(tabs) {
      var activeTab = tabs[0];
      chrome.tabs.sendMessage(activeTab.id, { message3: "false" });
    });
    intervalRun2 = true;
  }
};
