
# # Make this script executable with: chmod +x run_all.sh

# echo "🚀 Starting all hackathon servers..."

# # Start the Main Backend Server (Port 5000)
# echo "-> Starting Main Backend on port 5000"
# gnome-terminal --title="Main Backend" -- python3 backend_server.py &

# # Start the Simulator Server (Port 5001)
# echo "-> Starting Simulator on port 5001"
# gnome-terminal --title="Simulator" -- python3 simulator/simulator_server.py &

# # Start the Analytics Server (Port 5002)
# echo "-> Starting Analytics Server on port 5002"
# gnome-terminal --title="Analytics" -- python3 AnalyticsModule/analytics_server.py &

# # Start the Incident Logging Server (Port 5003)
# echo "-> Starting Incident Logger on port 5003"
# gnome-terminal --title="Incident Logger" -- python3 incident_logging/app.py &

# # Start the Learning Assistant Server (Port 5004)
# echo "-> Starting Learning Assistant on port 5004"
# gnome-terminal --title="Learning Assistant" -- python3 "python learning assisstant/assistant_server.py" &

# # NEW: Start the Companion Server (Port 5005)
# echo "-> Starting Companion Server on port 5005"
# gnome-terminal --title="Companion" -- python3 Companion/companion_server.py &


# echo "✅ All servers launched in separate terminals. Check the new windows."

#!/bin/bash

echo "🚀 Starting all hackathon servers in the background..."

# The '&' at the end of each line runs the command in the background
python3 backend_server.py &
python3 simulator/simulator_server.py &
python3 AnalyticsModule/analytics_server.py &
python3 incident_logging/app.py &
python3 python_learning_assisstant/assistant_server.py &
python3 Companion/companion_server.py &

echo "✅ All servers launched. Their logs will appear in this terminal."
echo "To stop them all, close this terminal window."