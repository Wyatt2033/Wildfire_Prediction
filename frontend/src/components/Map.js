import React, { useEffect, useState } from 'react';
import { MapContainer, GeoJSON } from 'react-leaflet';
import axios from 'axios';

const MapComponent = () => {
  const [mapData, setMapData] = useState({});

  useEffect(() => {
    axios.get('http://localhost:5000/api/country_fire_map')
      .then(response => {
        const data = response.data;
        for (let key in data) {
          try {
            data[key] = JSON.parse(data[key]);
          } catch (error) {
            console.error(`Error parsing JSON for key ${key}: `, error);
          }
        }
        setMapData(data);
      })
      .catch(error => {
        console.error('Error fetching data: ', error);
      });
  }, []);

  const style = (feature) => {
    return {
      fillColor: feature.properties.risk_color,
      weight: .5,
      opacity: 1,
      color: 'black',
      dashArray: '3',
      fillOpacity: 0.7
    };
  };

  return (
    <MapContainer style={{ height: "100vh", width: "100%" }} zoom={5} center={[37.8, -96.9]}>
      {Object.keys(mapData).map((key) => (
        <GeoJSON key={key} data={mapData[key]} style={style} />
      ))}
    </MapContainer>
  );
};

export default MapComponent;