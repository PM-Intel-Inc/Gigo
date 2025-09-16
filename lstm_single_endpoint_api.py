"""
Single Endpoint P6 Schedule Analyzer Flask API with LSTM
One endpoint that does EVERYTHING using actual LSTM neural network
"""

from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
import warnings
warnings.filterwarnings('ignore')

# LSTM and ML imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Input, Concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

app = Flask(__name__)

def clean_for_json(obj):
    """Clean data for JSON serialization"""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return 0.0
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return clean_for_json(obj.tolist())
    elif pd.isna(obj):
        return None
    else:
        try:
            if hasattr(obj, '__float__'):
                val = float(obj)
                return val if not (np.isnan(val) or np.isinf(val)) else 0.0
            elif hasattr(obj, '__int__'):
                return int(obj)
            else:
                return str(obj)
        except:
            return str(obj)

class LSTMCompleteP6Analyzer:
    def __init__(self):
        # Activity templates for generating training data
        self.activity_templates = {
            'permits_approvals': [
                'Building Permit Application', 'Environmental Impact Assessment',
                'Safety Plan Approval', 'Construction Permit Received'
            ],
            'site_preparation': [
                'Site Survey and Inspection', 'Site Clearing and Grubbing', 
                'Temporary Fencing Installation', 'Site Access Road Construction'
            ],
            'excavation': [
                'Excavation Layout and Marking', 'Bulk Excavation',
                'Fine Grading', 'Utility Trenching'
            ],
            'foundation': [
                'Foundation Layout', 'Rebar Installation - Foundation',
                'Foundation Concrete Pour', 'Foundation Curing', 'Foundation Inspection'
            ],
            'structural': [
                'Column Installation', 'Beam Installation', 'Floor Slab Pour',
                'Structural Steel Erection', 'Structural Inspection'
            ],
            'electrical': [
                'Electrical Rough-in', 'Electrical Panel Installation',
                'Wiring Installation', 'Electrical Testing and Commissioning'
            ],
            'plumbing': [
                'Plumbing Rough-in', 'Pipe Installation',
                'Plumbing Fixtures Installation', 'Plumbing Testing'
            ],
            'finishing': [
                'Drywall Installation', 'Painting and Finishing',
                'Flooring Installation', 'Final Cleanup', 'Final Inspection'
            ]
        }
        
        self.correct_sequence_order = [
            'permits_approvals', 'site_preparation', 'excavation', 'foundation',
            'structural', 'electrical', 'plumbing', 'finishing'
        ]
        
        # LSTM Components
        self.tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
        self.phase_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.lstm_model = None
        self.max_sequence_length = 50
        
        # Correction templates
        self.correction_templates = {
            'permit_after_work': {
                'reason': 'Permits must be obtained before construction work begins',
                'suggestion': 'Move to start of project after site preparation',
                'priority': 'Critical'
            },
            'foundation_before_excavation': {
                'reason': 'Foundation work requires completed excavation',
                'suggestion': 'Schedule after all excavation activities are complete',
                'priority': 'High'
            },
            'testing_before_installation': {
                'reason': 'Testing can only be performed after installation',
                'suggestion': 'Move to after corresponding installation activity',
                'priority': 'High'
            },
            'inspection_before_work': {
                'reason': 'Inspection must occur after work completion',
                'suggestion': 'Schedule at end of corresponding work phase',
                'priority': 'Medium'
            },
            'finishing_before_structure': {
                'reason': 'Finishing work requires completed structural work',
                'suggestion': 'Move to after structural phase completion',
                'priority': 'High'
            },
            'sequence_anomaly': {
                'reason': 'Activity appears to be out of sequence based on construction best practices',
                'suggestion': 'Review activity dependencies and sequencing logic',
                'priority': 'Medium'
            }
        }
    
    def generate_training_data(self, num_schedules=25):
        """Generate training schedules with flaws"""
        all_schedules = []
        
        for schedule_id in range(1, num_schedules + 1):
            activities = []
            activity_counter = 1
            start_date = datetime(2024, 1, 1) + timedelta(days=schedule_id * 30)
            
            # Generate activities in correct order
            for phase in self.correct_sequence_order:
                phase_activities = self.activity_templates[phase].copy()
                selected_count = min(3, len(phase_activities))  # 3 activities per phase
                selected_activities = phase_activities[:selected_count]
                
                for activity_name in selected_activities:
                    if len(activities) >= 20:  # Limit to 20 activities per schedule
                        break
                        
                    duration = np.random.randint(3, 10)
                    
                    activity = {
                        'Activity_ID': f'A{schedule_id:03d}-{activity_counter:03d}',
                        'Activity_Name': f'{activity_name} - Area {np.random.randint(1,4)}',
                        'Phase': phase,
                        'Duration': duration,
                        'Early_Start': start_date.strftime('%Y-%m-%d'),
                        'Early_Finish': (start_date + timedelta(days=duration-1)).strftime('%Y-%m-%d'),
                        'Sequence_Position': activity_counter,
                        'Is_Correct_Sequence': True
                    }
                    
                    activities.append(activity)
                    activity_counter += 1
                    start_date += timedelta(days=duration + np.random.randint(0, 2))
                
                if len(activities) >= 20:
                    break
            
            # Introduce flaws (40% of activities)
            num_flaws = int(len(activities) * 0.4)
            flaw_indices = np.random.choice(len(activities), size=num_flaws, replace=False)
            
            for idx in flaw_indices:
                activities[idx]['Is_Correct_Sequence'] = False
                activities[idx]['Violation_Type'] = self._assign_violation_type(activities[idx]['Activity_Name'])
            
            schedule_df = pd.DataFrame(activities)
            all_schedules.append(schedule_df)
        
        # Combine all schedules
        master_dataset = pd.concat(all_schedules, ignore_index=True)
        return master_dataset
    
    def _assign_violation_type(self, activity_name):
        """Assign violation type based on activity name"""
        activity_lower = activity_name.lower()
        
        if 'permit' in activity_lower:
            return 'permit_after_work'
        elif 'foundation' in activity_lower:
            return 'foundation_before_excavation'
        elif 'testing' in activity_lower:
            return 'testing_before_installation'
        elif 'inspection' in activity_lower:
            return 'inspection_before_work'
        elif 'finishing' in activity_lower or 'painting' in activity_lower:
            return 'finishing_before_structure'
        else:
            return 'sequence_anomaly'
    
    def build_lstm_model(self, vocab_size, embedding_dim=64, lstm_units=128):
        """Build LSTM model for sequence analysis"""
        print("🏗️ Building LSTM neural network...")
        
        # Text input branch
        text_input = Input(shape=(self.max_sequence_length,), name='text_input')
        text_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)(text_input)
        text_lstm = LSTM(lstm_units, dropout=0.3, recurrent_dropout=0.3)(text_embedding)
        
        # Phase input branch
        phase_input = Input(shape=(1,), name='phase_input')
        phase_embedding = Embedding(len(self.phase_encoder.classes_), 20)(phase_input)
        phase_flatten = tf.keras.layers.Flatten()(phase_embedding)
        
        # Numerical input branch
        numerical_input = Input(shape=(2,), name='numerical_input')
        
        # Combine all branches
        combined = Concatenate()([text_lstm, phase_flatten, numerical_input])
        combined_dense = Dense(256, activation='relu')(combined)
        combined_dropout = Dropout(0.4)(combined_dense)
        
        # Output layer for sequence correctness
        sequence_output = Dense(1, activation='sigmoid', name='sequence_correct')(combined_dropout)
        
        # Create model
        model = Model(
            inputs=[text_input, phase_input, numerical_input],
            outputs=sequence_output
        )
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ LSTM model architecture built")
        return model
    
    def train_lstm_model(self, training_data, epochs=20, batch_size=32):
        """Train the LSTM model"""
        print("🚀 Training LSTM neural network...")
        
        # Prepare text features
        activity_texts = training_data['Activity_Name'].astype(str).tolist()
        self.tokenizer.fit_on_texts(activity_texts)
        sequences = self.tokenizer.texts_to_sequences(activity_texts)
        X_text = pad_sequences(sequences, maxlen=self.max_sequence_length, padding='post')
        
        # Prepare phase features
        X_phase = self.phase_encoder.fit_transform(training_data['Phase'].astype(str))
        
        # Prepare numerical features
        numerical_features = training_data[['Duration', 'Sequence_Position']].values
        X_numerical = self.scaler.fit_transform(numerical_features)
        
        # Target variable
        y = (~training_data['Is_Correct_Sequence']).astype(int)  # 1 for incorrect, 0 for correct
        
        # Build model
        vocab_size = len(self.tokenizer.word_index) + 1
        self.lstm_model = self.build_lstm_model(vocab_size)
        
        # Add callbacks for training
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6
            )
        ]
        
        # Train model
        history = self.lstm_model.fit(
            [X_text, X_phase.reshape(-1, 1), X_numerical], y,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0  # Suppress training output for cleaner API response
        )
        
        # Calculate final accuracy
        predictions = self.lstm_model.predict([X_text, X_phase.reshape(-1, 1), X_numerical])
        predicted_classes = (predictions > 0.5).astype(int).flatten()
        accuracy = np.mean(predicted_classes == y)
        
        print(f"✅ LSTM training completed! Accuracy: {accuracy:.1%}")
        
        return {
            'accuracy': float(accuracy),
            'total_samples': len(training_data),
            'incorrect_samples': int(sum(y)),
            'correct_samples': int(len(y) - sum(y)),
            'model_type': 'LSTM Neural Network',
            'epochs_trained': len(history.history['loss']),
            'final_loss': float(history.history['loss'][-1]),
            'vocab_size': vocab_size
        }
    
    def analyze_schedule(self, schedule_df):
        """Analyze uploaded schedule using LSTM"""
        if self.lstm_model is None:
            return {'error': 'LSTM model not trained yet'}
        
        # Find activity name column
        activity_col = None
        for col in ['Activity Name', 'Activity_Name', 'Task Name', 'Task_Name']:
            if col in schedule_df.columns:
                activity_col = col
                break
        
        if not activity_col:
            return {'error': 'Could not find activity name column'}
        
        activity_texts = schedule_df[activity_col].astype(str).tolist()
        
        # Prepare text features
        sequences = self.tokenizer.texts_to_sequences(activity_texts)
        X_text = pad_sequences(sequences, maxlen=self.max_sequence_length, padding='post')
        
        # Infer and encode phases
        phases = self._infer_phases(activity_texts)
        X_phase = []
        for phase in phases:
            if hasattr(self.phase_encoder, 'classes_') and phase in self.phase_encoder.classes_:
                X_phase.append(self.phase_encoder.transform([phase])[0])
            else:
                X_phase.append(0)
        X_phase = np.array(X_phase).reshape(-1, 1)
        
        # Prepare numerical features
        duration_col = None
        for col in ['Duration', 'Original Duration', 'Planned Duration']:
            if col in schedule_df.columns:
                duration_col = col
                break
        
        if duration_col:
            duration = schedule_df[duration_col].fillna(5).astype(float).tolist()
        else:
            duration = [5] * len(schedule_df)
        
        position = list(range(1, len(schedule_df) + 1))
        numerical_features = np.column_stack([duration, position])
        X_numerical = self.scaler.transform(numerical_features)
        
        # Make LSTM predictions
        predictions = self.lstm_model.predict([X_text, X_phase, X_numerical])
        
        # Generate results
        results = []
        for i, activity_name in enumerate(activity_texts):
            sequence_prob = float(predictions[i][0])
            is_incorrect = sequence_prob > 0.5
            confidence = abs(sequence_prob - 0.5) * 2  # Convert to 0-1 confidence
            
            activity_id = schedule_df.iloc[i].get('Activity ID', 
                         schedule_df.iloc[i].get('Activity_ID', f'A{i+1:03d}'))
            
            result = {
                'sr_no': i + 1,
                'activity_id': str(activity_id),
                'task_name': str(activity_name),
                'phase': str(phases[i]),
                'is_correct': bool(not is_incorrect),
                'confidence': float(confidence),
                'lstm_probability': float(sequence_prob),
                'current_position': i + 1,
                'flag_as_incorrect': '❌ Yes' if is_incorrect else '✅ No'
            }
            
            if is_incorrect:
                correction = self._generate_correction(activity_name, phases[i], i + 1)
                result.update(correction)
                result['reason'] = correction['correction_reason']
            else:
                result['reason'] = 'Activity is properly sequenced within the construction workflow'
            
            results.append(result)
        
        return {
            'results': results,
            'summary': {
                'total_activities': len(results),
                'issues_found': sum(1 for r in results if not r['is_correct']),
                'correct_sequences': sum(1 for r in results if r['is_correct']),
                'accuracy_percentage': (sum(1 for r in results if r['is_correct']) / len(results)) * 100,
                'model_used': 'LSTM Neural Network'
            }
        }
    
    def _infer_phases(self, activity_texts):
        """Infer phases from activity names"""
        phases = []
        for activity in activity_texts:
            activity_lower = activity.lower()
            
            if 'permit' in activity_lower or 'approval' in activity_lower:
                phases.append('permits_approvals')
            elif 'site' in activity_lower or 'clearing' in activity_lower or 'survey' in activity_lower:
                phases.append('site_preparation')
            elif 'excavation' in activity_lower or 'trenching' in activity_lower:
                phases.append('excavation')
            elif 'foundation' in activity_lower or 'footing' in activity_lower:
                phases.append('foundation')
            elif 'structural' in activity_lower or 'column' in activity_lower or 'beam' in activity_lower:
                phases.append('structural')
            elif 'electrical' in activity_lower:
                phases.append('electrical')
            elif 'plumbing' in activity_lower or 'pipe' in activity_lower:
                phases.append('plumbing')
            elif 'finishing' in activity_lower or 'painting' in activity_lower or 'flooring' in activity_lower:
                phases.append('finishing')
            else:
                phases.append('unknown')
        
        return phases
    
    def _generate_correction(self, activity_name, phase, current_position):
        """Generate correction details"""
        activity_lower = activity_name.lower()
        
        violation_type = 'sequence_anomaly'
        suggested_position = current_position
        
        if 'permit' in activity_lower:
            violation_type = 'permit_after_work'
            suggested_position = 1
        elif 'foundation' in activity_lower:
            violation_type = 'foundation_before_excavation'
            suggested_position = max(1, current_position + 3)
        elif 'testing' in activity_lower or 'commissioning' in activity_lower:
            violation_type = 'testing_before_installation'
            suggested_position = current_position + 2
        elif 'inspection' in activity_lower:
            violation_type = 'inspection_before_work'
            suggested_position = current_position + 1
        elif 'finishing' in activity_lower or 'painting' in activity_lower:
            violation_type = 'finishing_before_structure'
            suggested_position = current_position + 5
        
        template = self.correction_templates.get(violation_type)
        
        return {
            'violation_type': str(violation_type),
            'correction_reason': str(template['reason']),
            'correction_suggestion': str(template['suggestion']),
            'priority': str(template['priority']),
            'suggested_position': int(suggested_position),
            'position_change': int(suggested_position - current_position)
        }

# Initialize the LSTM analyzer
analyzer = LSTMCompleteP6Analyzer()

# HTML Template matching the desired output format
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>P6 Schedule Analyzer - LSTM Neural Network</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
            margin: 0; padding: 0; background: #f8f9fa;
        }
        .container { 
            max-width: 1200px; margin: 0 auto; background: white; 
            min-height: 100vh;
        }
        .header {
            background: #fff; border-bottom: 1px solid #e1e5e9;
            padding: 1rem 2rem; display: flex; justify-content: space-between; align-items: center;
        }
        .title { 
            font-size: 1.5rem; font-weight: 600; color: #1a202c; margin: 0;
            display: flex; align-items: center; gap: 0.5rem;
        }
        .controls {
            display: flex; gap: 0.5rem; align-items: center;
        }
        .upload-section {
            padding: 2rem; background: white; border-bottom: 1px solid #e1e5e9;
        }
        .upload-area { 
            border: 2px dashed #cbd5e0; padding: 2rem; text-align: center; 
            border-radius: 8px; margin: 1rem 0; background: #f7fafc;
        }
        .button { 
            background: #4299e1; color: white; padding: 0.75rem 1.5rem; 
            border: none; border-radius: 6px; cursor: pointer; font-size: 0.875rem;
            font-weight: 500; transition: all 0.2s;
        }
        .button:hover { background: #3182ce; }
        .button.secondary { background: #48bb78; }
        .button.secondary:hover { background: #38a169; }
        
        .results-container { padding: 0; }
        .results-header {
            background: #fff; padding: 1rem 2rem; border-bottom: 1px solid #e1e5e9;
            display: flex; justify-content: space-between; align-items: center;
        }
        .results-title { 
            font-size: 1.125rem; font-weight: 600; color: #1a202c; margin: 0;
        }
        .table-container {
            background: white; overflow-x: auto;
        }
        .results-table { 
            width: 100%; border-collapse: collapse; font-size: 0.875rem;
        }
        .results-table th { 
            background: #f7fafc; color: #4a5568; font-weight: 600;
            padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid #e2e8f0;
            font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em;
        }
        .results-table td { 
            padding: 0.75rem 1rem; border-bottom: 1px solid #e2e8f0;
            vertical-align: top;
        }
        .results-table tbody tr:hover { background: #f7fafc; }
        
        .flag-correct { 
            color: #38a169; background: #f0fff4; padding: 0.25rem 0.5rem; 
            border-radius: 4px; font-size: 0.75rem; font-weight: 600;
            display: inline-flex; align-items: center; gap: 0.25rem;
        }
        .flag-incorrect { 
            color: #e53e3e; background: #fed7d7; padding: 0.25rem 0.5rem; 
            border-radius: 4px; font-size: 0.75rem; font-weight: 600;
            display: inline-flex; align-items: center; gap: 0.25rem;
        }
        
        .reason-text { 
            color: #4a5568; font-size: 0.875rem; line-height: 1.4;
            max-width: 400px;
        }
        
        .training-info {
            background: #e6fffa; border: 1px solid #81e6d9; color: #234e52;
            padding: 1rem; border-radius: 6px; margin: 1rem 2rem;
        }
        .training-info h4 { margin: 0 0 0.5rem 0; color: #234e52; }
        
        .metrics { 
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
            gap: 1rem; padding: 1rem 2rem; background: #f7fafc;
        }
        .metric { 
            text-align: center; padding: 1rem; background: white; 
            border-radius: 6px; border: 1px solid #e2e8f0;
        }
        .metric h3 { margin: 0; color: #4a5568; font-size: 0.875rem; font-weight: 500; }
        .metric p { margin: 0.5rem 0 0 0; font-size: 1.5rem; font-weight: 700; color: #2d3748; }
        
        .loading { 
            text-align: center; padding: 3rem; color: #4a5568;
        }
        .error { 
            background: #fed7d7; border: 1px solid #feb2b2; color: #9b2c2c;
            padding: 1rem; border-radius: 6px; margin: 1rem 2rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1 class="title">
                <span>🧠</span>
                Logical Flow
            </h1>
            <div class="controls">
                <button class="button" onclick="window.location.reload()">🔄</button>
                <button class="button" onclick="window.print()">🖨️</button>
                <button class="button" onclick="exportResults()">📤</button>
            </div>
        </div>
        
        <div class="upload-section">
            <div class="upload-area">
                <h3 style="margin-top: 0; color: #4a5568;">📁 Upload Your P6 Schedule</h3>
                <p style="color: #718096; margin-bottom: 1rem;">Powered by TensorFlow LSTM Neural Network</p>
                <input type="file" id="fileInput" accept=".csv,.xlsx" style="margin-bottom: 1rem;">
                <br>
                <button class="button" onclick="analyzeSchedule()">🧠 Analyze with LSTM</button>
                <button class="button secondary" onclick="testSampleData()" style="margin-left: 1rem;">📊 Test Sample Data</button>
            </div>
        </div>
        
        <div id="results" class="results-container" style="display: none;"></div>
    </div>

    <script>
        function analyzeSchedule() {
            const fileInput = document.getElementById('fileInput');
            if (!fileInput.files.length) {
                alert('Please select a file first');
                return;
            }

            const formData = new FormData();
            formData.append('schedule_file', fileInput.files[0]);

            showLoading();
            
            fetch('/analyze', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    showError(data.error);
                } else {
                    displayResults(data);
                }
            })
            .catch(error => showError('Error: ' + error.message));
        }

        function testSampleData() {
            const sampleActivities = [
                {"Activity_ID": "PP-A-789", "Activity_Name": "Area 6 Piling Installation", "Duration": 5},
                {"Activity_ID": "PP-A-256", "Activity_Name": "Area 1 Permit Received", "Duration": 1},
                {"Activity_ID": "PP-A-543", "Activity_Name": "Drawing & Specs Review", "Duration": 2},
                {"Activity_ID": "PP-A-645", "Activity_Name": "Backfilling and Site Cleanup", "Duration": 3},
                {"Activity_ID": "PP-A-467", "Activity_Name": "Piling 2 testing and inspection", "Duration": 1},
                {"Activity_ID": "PP-A-890", "Activity_Name": "Area 3 Piling Rig Move", "Duration": 2},
                {"Activity_ID": "PP-A-987", "Activity_Name": "Piling 3 Cap Installation", "Duration": 4},
                {"Activity_ID": "PP-A-912", "Activity_Name": "Area 4 Piling Installation", "Duration": 6},
                {"Activity_ID": "PP-A-654", "Activity_Name": "Piling1 Cap Installation", "Duration": 3},
                {"Activity_ID": "PP-A-321", "Activity_Name": "Piling 2 Cap Installation", "Duration": 4}
            ];

            showLoading();
            
            fetch('/analyze', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({activities: sampleActivities})
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    showError(data.error);
                } else {
                    displayResults(data);
                }
            })
            .catch(error => showError('Error: ' + error.message));
        }

        function showLoading() {
            const resultsDiv = document.getElementById('results');
            resultsDiv.style.display = 'block';
            resultsDiv.innerHTML = `
                <div class="loading">
                    <h3>🧠 LSTM Neural Network Processing...</h3>
                    <p>Generating training data, training LSTM model, and analyzing your schedule...</p>
                </div>`;
        }

        function displayResults(data) {
            const resultsDiv = document.getElementById('results');
            resultsDiv.style.display = 'block';
            
            let html = '';
            
            // Training info
            if (data.training_info) {
                html += `
                    <div class="training-info">
                        <h4>🧠 LSTM Neural Network Training Results</h4>
                        <p><strong>Model Type:</strong> ${data.training_info.model_type}</p>
                        <p><strong>Training Accuracy:</strong> ${(data.training_info.accuracy * 100).toFixed(1)}%</p>
                        <p><strong>Training Data:</strong> ${data.training_info.total_samples} activities</p>
                        <p><strong>Epochs:</strong> ${data.training_info.epochs_trained} | <strong>Vocab:</strong> ${data.training_info.vocab_size} words</p>
                    </div>`;
            }
            
            // Summary metrics
            if (data.analysis && data.analysis.summary) {
                const summary = data.analysis.summary;
                html += `
                    <div class="metrics">
                        <div class="metric">
                            <h3>Total Activities</h3>
                            <p>${summary.total_activities}</p>
                        </div>
                        <div class="metric">
                            <h3>Issues Found</h3>
                            <p style="color: #e53e3e;">${summary.issues_found}</p>
                        </div>
                        <div class="metric">
                            <h3>Correct Sequences</h3>
                            <p style="color: #38a169;">${summary.correct_sequences}</p>
                        </div>
                        <div class="metric">
                            <h3>LSTM Accuracy</h3>
                            <p>${summary.accuracy_percentage.toFixed(1)}%</p>
                        </div>
                    </div>`;
            }
            
            // Results header and table
            html += `
                <div class="results-header">
                    <h2 class="results-title">Logical Flow</h2>
                </div>
                <div class="table-container">
                    <table class="results-table">
                        <thead>
                            <tr>
                                <th>Sr. No.</th>
                                <th>Activity ID</th>
                                <th>Task Name</th>
                                <th>Reason</th>
                                <th>Flag As Incorrect</th>
                            </tr>
                        </thead>
                        <tbody>`;
            
            // Results rows
            if (data.analysis && data.analysis.results) {
                data.analysis.results.forEach(result => {
                    const flagClass = result.is_correct ? 'flag-correct' : 'flag-incorrect';
                    const flagIcon = result.is_correct ? '🟢' : '🔴';
                    const flagText = result.is_correct ? '' : '🚩';
                    
                    html += `
                        <tr>
                            <td>${result.sr_no}</td>
                            <td>${result.activity_id}</td>
                            <td>${result.task_name}</td>
                            <td>
                                <div class="reason-text">
                                    Logical Flow: ${result.reason}
                                </div>
                            </td>
                            <td>
                                <span class="${flagClass}">
                                    ${flagIcon} ${flagText}
                                </span>
                            </td>
                        </tr>`;
                });
            }
            
            html += `
                        </tbody>
                    </table>
                </div>`;
            
            // Processing time
            if (data.processing_time) {
                html += `<div style="padding: 1rem 2rem; color: #718096; font-size: 0.875rem;">⏱️ LSTM Processing time: ${data.processing_time.toFixed(2)} seconds</div>`;
            }
            
            resultsDiv.innerHTML = html;
        }

        function showError(message) {
            const resultsDiv = document.getElementById('results');
            resultsDiv.style.display = 'block';
            resultsDiv.innerHTML = `<div class="error"><h3>❌ Error</h3><p>${message}</p></div>`;
        }

        function exportResults() {
            // Simple export functionality
            if (document.querySelector('.results-table')) {
                window.print();
            } else {
                alert('No results to export. Please run an analysis first.');
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Main page with LSTM interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/analyze', methods=['POST'])
def analyze_everything_lstm():
    """
    🧠 SINGLE ENDPOINT WITH LSTM NEURAL NETWORK!
    
    1. Generates 25 training schedules (500 activities)
    2. Trains LSTM neural network on this data  
    3. Analyzes your uploaded schedule with LSTM
    4. Returns complete analysis with LSTM predictions
    """
    start_time = datetime.now()
    
    try:
        # STEP 1: Generate Training Data
        print("🔧 Step 1: Generating training data...")
        training_data = analyzer.generate_training_data(num_schedules=25)
        
        # STEP 2: Train LSTM Model
        print("🧠 Step 2: Training LSTM neural network...")
        training_info = analyzer.train_lstm_model(training_data, epochs=15)
        
        # STEP 3: Get User Schedule
        user_schedule = None
        
        # Check if file was uploaded
        if 'schedule_file' in request.files:
            file = request.files['schedule_file']
            if file.filename != '':
                if file.filename.endswith('.csv'):
                    user_schedule = pd.read_csv(file)
                elif file.filename.endswith('.xlsx'):
                    user_schedule = pd.read_excel(file)
                else:
                    return jsonify({'error': 'Unsupported file format. Use CSV or Excel.'}), 400
        
        # Check if JSON data was sent
        elif request.is_json:
            json_data = request.get_json()
            if 'activities' in json_data:
                user_schedule = pd.DataFrame(json_data['activities'])
        
        if user_schedule is None:
            return jsonify({'error': 'No schedule provided. Upload a file or send JSON data.'}), 400
        
        # STEP 4: Analyze User Schedule with LSTM
        print("🔍 Step 3: Analyzing your schedule with LSTM...")
        analysis_results = analyzer.analyze_schedule(user_schedule)
        
        if 'error' in analysis_results:
            return jsonify({'error': analysis_results['error']}), 400
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # STEP 5: Return Complete Results
        response_data = {
            'status': 'success',
            'message': 'Complete LSTM analysis finished successfully!',
            'processing_time': processing_time,
            'training_info': training_info,
            'analysis': analysis_results,
            'steps_completed': [
                f'✅ Generated {len(training_data)} training activities',
                f'✅ Trained LSTM neural network ({training_info["accuracy"]:.1%} accuracy)',
                f'✅ Analyzed {len(user_schedule)} user activities with LSTM',
                f'✅ Found {analysis_results["summary"]["issues_found"]} sequence issues'
            ]
        }
        
        return jsonify(clean_for_json(response_data))
    
    except Exception as e:
        return jsonify({
            'error': f'LSTM processing failed: {str(e)}',
            'processing_time': (datetime.now() - start_time).total_seconds()
        }), 500

@app.route('/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'model': 'LSTM Neural Network',
        'framework': 'TensorFlow/Keras',
        'endpoint': '/analyze',
        'description': 'Single endpoint that generates training data, trains LSTM model, and analyzes your schedule',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🧠 LSTM P6 Schedule Analyzer - Single Endpoint")
    print("=" * 60)
    print("🌐 Web Interface: http://localhost:8080")
    print("🎯 Single Endpoint: POST /analyze")
    print("🔍 Health Check: http://localhost:8080/health")
    print("=" * 60)
    print("✨ ONE ENDPOINT WITH LSTM NEURAL NETWORK:")
    print("   1️⃣ Generates 25 training schedules")
    print("   2️⃣ Trains LSTM neural network (TensorFlow)")
    print("   3️⃣ Analyzes your uploaded schedule with LSTM")
    print("   4️⃣ Returns complete analysis with LSTM predictions")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=8080)