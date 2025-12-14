import React, { useEffect, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

// Component to fetch and display the physics data
const FractalChart = () => {
    const [data, setData] = useState([]);

    useEffect(() => {
        // Fetch data from static folder
        fetch('/data/simulation_data.json')
            .then((response) => response.json())
            .then((jsonData) => setData(jsonData))
            .catch((error) => console.error('Error loading fractal data:', error));
    }, []);

    if (!data || data.length === 0) {
        return <div>Loading Physics Engine...</div>;
    }

    return (
        <div style={{ width: '100%', height: 400, background: '#000', borderRadius: '8px', padding: '20px' }}>
            <h3 style={{ color: '#d4af37', textAlign: 'center' }}>7D Phi-Resonance Simulation (Live C++ Data)</h3>
            <ResponsiveContainer width="100%" height="100%">
                <LineChart
                    data={data}
                    margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                    <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                    <XAxis dataKey="step" stroke="#666" />
                    <YAxis stroke="#666" />
                    <Tooltip contentStyle={{ backgroundColor: '#222', borderColor: '#d4af37' }} />
                    <Legend />
                    {/* We plot the first 3 dimensions for clarity */}
                    <Line type="monotone" dataKey="d0" stroke="#ff0000" name="Dim 1 (Time)" dot={false} />
                    <Line type="monotone" dataKey="d1" stroke="#00ff00" name="Dim 2 (Space)" dot={false} />
                    <Line type="monotone" dataKey="d2" stroke="#0000ff" name="Dim 3 (Matter)" dot={false} />
                    <Line type="monotone" dataKey="d3" stroke="#d4af37" name="Dim 4 (Phi)" dot={false} />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
};

export default FractalChart;
