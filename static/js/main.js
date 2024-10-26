// Fetch memory usage data from the Flask backend
function fetchMemoryData() {
    fetch("/memory_data")
        .then(response => response.json())
        .then(data => {
            document.getElementById("memory-data").innerText = data.memory_usage;
        })
        .catch(error => console.error("Error fetching memory data:", error));
}

// Fetch data every 5 seconds
setInterval(fetchMemoryData, 5000);
