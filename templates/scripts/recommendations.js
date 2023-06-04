function getRecommendations() {
    // Get the username from the input field
    var username = document.getElementById("username").value;
    // Create a new XMLHttpRequest object
    var xhr = new XMLHttpRequest();
    // Set the request method and URL
    xhr.open("GET", "/recommendations?username=" + username);
    // Send the request
    xhr.send();
    // Handle the response
    xhr.onload = function() {
    // Check if the request was successful
    if (xhr.status === 200) {
    // Get the recommendations from the response
    var recommendations = JSON.parse(xhr.responseText);
    // Display the recommendations
    var ul = document.getElementById("recommendations");
    for (var i = 0; i < recommendations.length; i++) {
    var li = document.createElement("li");
    li.textContent = recommendations[i];
    ul.appendChild(li);
    }
    } else {
    // Display an error message
    alert("An error occurred.");
    }
    };
    }
    