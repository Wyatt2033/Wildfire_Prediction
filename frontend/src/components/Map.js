import React, { useEffect, useState } from 'react';
import { MapContainer, GeoJSON } from 'react-leaflet';
import axios from 'axios';

const MapComponent = () => {
  const [mapData, setMapData] = useState({});

useEffect(() => {
  console.log('Fetching data from API...');
  axios.get('http://localhost:5000/api/country_fire_map')
    .then(response => {
      console.log('Data fetched successfully:', response.data);
      // Parse the string data into JSON
      const parsedData = {};
      for (const key in response.data) {
        parsedData[key] = JSON.parse(response.data[key]);
      }
      setMapData(parsedData);
    })
    .catch(error => {
      console.error('Error fetching data: ', error);
    });
}, []);

  const style = (feature, layerType) => {
    const isState = layerType === 'state';
    return {
      fillColor: feature.properties.risk_color,
      weight: isState ? 5 : 2, // Use a thicker weight for states
      opacity: 1,
      color: isState ? 'black' : 'white', // Use black for states, white for counties
      dashArray: '3',
      fillOpacity: 0.7
    };
  };


  console.log('Rendering MapComponent with data:', mapData); // Added console output


  return (
    <MapContainer style={{ height: "100vh", width: "100%" }} zoom={5} center={[37.8, -96.9]}>
      {mapData.us && <GeoJSON key={'us'} data={mapData.us} style={(feature) => style(feature, 'county')} />}
      {mapData.states && <GeoJSON key={'states'} data={mapData.states} style={(feature) => style(feature, 'state')} />}
    </MapContainer>
  );
};

export default MapComponent;