const express = require('express');
const cors = require('cors'); // Import cors module
const axios = require('axios');
const {spawn} = require('child_process');
const app = express();
const port = 3001;

var corsOptions = {
  origin: 'http://localhost:3000', // replace with the origin of your React app
  optionsSuccessStatus: 200
}

app.use(cors(corsOptions)); // Use cors as middleware

app.get('/api/mapdata', (req, res) => {
  // Replace this with your actual data
  const data = [
    {
      coordinates: [[37.7749, -122.4194]],
      risk_color: 'red'
    },
    // Add more data as needed
  ];

  res.json(data);
});

// New route to act as a proxy to your Flask API
app.get('/api/country_fire_map', (req, res) => {
  axios.get('http://localhost:5000/api/country_fire_map')
    .then(response => {
      res.json(response.data);
    })
    .catch(error => {
      console.error('Error fetching data: ', error);
      res.status(500).json({ error: 'An error occurred while fetching data' });
    });
});

app.get('/api/countrymapdata', (req, res) => {
  const python = spawn('python3', ['./geocoding.py']);
  let dataToSend;

  python.stdout.on('data', function (data) {
    console.log('Pipe data from python script ...');
    dataToSend = data.toString();
  });

  python.on('close', (code) => {
    console.log(`child process close all stdio with code ${code}`);
    res.send(dataToSend)
  });
});

app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});