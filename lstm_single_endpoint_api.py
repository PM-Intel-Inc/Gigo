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
        # Comprehensive construction sequence phases (in correct order)
        self.construction_sequence = [
            'permits_approvals',
            'site_preparation', 
            'demolition',
            'surveying',
            'excavation',
            'shoring',
            'dewatering',
            'piling',
            'blinding',
            'waterproofing_below',
            'formwork',
            'rebar',
            'foundation',
            'backfilling',
            'structural_steel',
            'structural_concrete',
            'masonry',
            'roofing',
            'waterproofing_above',
            'curtain_wall',
            'mep_rough',
            'insulation',
            'drywall',
            'flooring',
            'painting',
            'mep_finish',
            'fixtures',
            'testing',
            'commissioning',
            'landscaping',
            'final_inspection'
        ]
        
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
        
        # Comprehensive correction templates for all construction sequences
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
            'formwork_before_excavation': {
                'reason': 'Formwork and shuttering requires completed excavation',
                'suggestion': 'Schedule after excavation but before concrete pouring',
                'priority': 'High'
            },
            'backfilling_before_foundation': {
                'reason': 'Backfilling can only occur after foundation work is complete',
                'suggestion': 'Schedule after all foundation and concrete work',
                'priority': 'High'
            },
            'hvac_before_structure': {
                'reason': 'HVAC ducting installation requires completed structural work',
                'suggestion': 'Schedule after structural phase and before finishing work',
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
    
    def analyze_schedule(self, schedule_df, max_batch_size=50):
        """Analyze uploaded schedule using LSTM with batching for large schedules"""
        if self.lstm_model is None:
            return {'error': 'LSTM model not trained yet'}
        
        # Check schedule size
        num_activities = len(schedule_df)
        is_large_schedule = num_activities > 100
        
        if num_activities > 1000:
            return {
                'error': f'Schedule too large: {num_activities} activities',
                'message': 'Maximum supported schedule size is 1000 activities. Please split your schedule into smaller sections.',
                'suggestion': 'Consider analyzing critical path activities or breaking the schedule into phases.'
            }
        
        # Find activity name column
        activity_col = None
        for col in ['Activity Name', 'Activity_Name', 'Task Name', 'Task_Name']:
            if col in schedule_df.columns:
                activity_col = col
                break
        
        if not activity_col:
            return {'error': 'Could not find activity name column'}
        
        activity_texts_raw = schedule_df[activity_col].astype(str).tolist()
        
        # Check for duplicate activity names
        duplicates = self._check_duplicates(schedule_df, activity_col)
        
        # Handle duplicates by renaming them instead of rejecting
        activity_texts = []
        duplicate_counter = {}
        
        for i, activity in enumerate(activity_texts_raw):
            if activity in duplicate_counter:
                # This is a duplicate - add a counter to make it unique
                duplicate_counter[activity] += 1
                renamed_activity = f"{activity} #{duplicate_counter[activity]}"
                activity_texts.append(renamed_activity)
            else:
                # First occurrence or unique activity
                duplicate_counter[activity] = 1
                activity_texts.append(activity)
        
        # Store duplicate warning info for later inclusion in results
        duplicate_warning = None
        if duplicates['has_duplicates']:
            duplicate_warning = {
                'has_duplicates': True,
                'message': 'Warning: Duplicate activity names detected and automatically numbered',
                'duplicates': duplicates['duplicates'],
                'total_duplicates': duplicates['duplicate_count']
            }
        
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
        
        # Make LSTM predictions with batching for large schedules
        if num_activities > 100:
            # Process in batches to avoid memory issues
            predictions = []
            batch_size = max_batch_size
            for i in range(0, num_activities, batch_size):
                end_idx = min(i + batch_size, num_activities)
                batch_predictions = self.lstm_model.predict(
                    [X_text[i:end_idx], X_phase[i:end_idx], X_numerical[i:end_idx]],
                    verbose=0,
                    batch_size=32
                )
                predictions.append(batch_predictions)
            predictions = np.vstack(predictions)
        else:
            predictions = self.lstm_model.predict([X_text, X_phase, X_numerical], verbose=0)
        
        # Perform additional sequence validation
        sequence_issues = self._validate_construction_sequence(activity_texts, phases)
        
        # Generate results
        results = []
        for i, activity_name in enumerate(activity_texts):
            sequence_prob = float(predictions[i][0])
            
            # Check both LSTM prediction and rule-based validation
            lstm_thinks_incorrect = sequence_prob > 0.5
            rule_based_issue = i in sequence_issues
            is_incorrect = lstm_thinks_incorrect or rule_based_issue
            
            confidence = abs(sequence_prob - 0.5) * 2  # Convert to 0-1 confidence
            if rule_based_issue:
                confidence = max(confidence, 0.8)  # High confidence for rule-based issues
            
            activity_id = schedule_df.iloc[i].get('Activity ID', 
                         schedule_df.iloc[i].get('Activity_ID', f'A{i+1:03d}'))
            
            # Get original activity name (before duplicate renaming)
            original_activity = activity_texts_raw[i]
            display_name = activity_name  # This includes the #2, #3 suffix for duplicates
            
            result = {
                'sr_no': i + 1,
                'activity_id': str(activity_id),
                'task_name': str(display_name),
                'original_name': str(original_activity),
                'is_duplicate': display_name != original_activity,
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
                if rule_based_issue and i in sequence_issues:
                    result['reason'] = sequence_issues[i].get('reason', result['reason'])
            else:
                result['reason'] = 'Activity is properly sequenced within the construction workflow'
            
            # Add duplicate indicator to reason if it's a duplicate
            if result['is_duplicate']:
                result['reason'] = f"[DUPLICATE ACTIVITY] {result['reason']}"
            
            results.append(result)
        
        # For large schedules, add pagination info
        if num_activities > 100:
            # Only return first 100 results for display, but analyze all
            display_results = results[:100]
            response = {
                'results': display_results,
                'all_results_count': len(results),
                'displayed_count': len(display_results),
                'is_truncated': True,
                'summary': {
                    'total_activities': len(results),
                    'issues_found': sum(1 for r in results if not r['is_correct']),
                    'correct_sequences': sum(1 for r in results if r['is_correct']),
                    'accuracy_percentage': (sum(1 for r in results if r['is_correct']) / len(results)) * 100,
                    'model_used': 'LSTM Neural Network',
                    'schedule_size': 'Large' if num_activities > 200 else 'Medium'
                }
            }
        else:
            response = {
                'results': results,
                'summary': {
                    'total_activities': len(results),
                    'issues_found': sum(1 for r in results if not r['is_correct']),
                    'correct_sequences': sum(1 for r in results if r['is_correct']),
                    'accuracy_percentage': (sum(1 for r in results if r['is_correct']) / len(results)) * 100,
                    'model_used': 'LSTM Neural Network'
                }
            }
        
        # Add duplicate warning if duplicates were found
        if duplicate_warning:
            response['duplicate_warning'] = duplicate_warning
            
        return response
    
    def _validate_construction_sequence(self, activity_texts, phases):
        """Comprehensive validation of construction sequence based on industry best practices"""
        issues = {}
        
        # Define the master sequence order (from earliest to latest)
        sequence_priority = {
            'permits_approvals': 1,
            'site_preparation': 2,
            'demolition': 3,
            'surveying': 4,
            'excavation': 5,
            'shoring': 6,
            'dewatering': 7,
            'piling': 8,
            'blinding': 9,
            'waterproofing_below': 10,
            'formwork': 11,
            'rebar': 12,
            'foundation': 13,
            'backfilling': 14,
            'structural_steel': 15,
            'structural_concrete': 16,
            'masonry': 17,
            'roofing': 18,
            'waterproofing_above': 19,
            'curtain_wall': 20,
            'mep_rough': 21,
            'insulation': 22,
            'drywall': 23,
            'flooring': 24,
            'painting': 25,
            'mep_finish': 26,
            'fixtures': 27,
            'testing': 28,
            'commissioning': 29,
            'landscaping': 30,
            'final_inspection': 31,
            'unknown': 99
        }
        
        # Track phase positions for dependency checking
        phase_positions = {}
        for i, phase in enumerate(phases):
            if phase not in phase_positions:
                phase_positions[phase] = []
            phase_positions[phase].append(i)
        
        # Define critical dependencies (phase -> must come after these phases)
        critical_dependencies = {
            'foundation': ['excavation', 'piling', 'formwork', 'rebar'],
            'formwork': ['excavation', 'piling', 'blinding'],
            'rebar': ['formwork'],
            'backfilling': ['foundation', 'waterproofing_below'],
            'structural_concrete': ['foundation', 'structural_steel'],
            'structural_steel': ['foundation'],
            'masonry': ['structural_concrete', 'structural_steel'],
            'roofing': ['structural_concrete', 'structural_steel'],
            'curtain_wall': ['structural_concrete', 'structural_steel'],
            'mep_rough': ['structural_concrete', 'drywall'],
            'mep_finish': ['mep_rough', 'drywall'],
            'drywall': ['mep_rough', 'insulation'],
            'flooring': ['drywall'],
            'painting': ['drywall'],
            'waterproofing_below': ['excavation', 'blinding'],
            'waterproofing_above': ['structural_concrete', 'roofing'],
            'insulation': ['mep_rough'],
            'testing': ['mep_finish'],
            'commissioning': ['testing'],
            'landscaping': ['backfilling'],
            'final_inspection': ['commissioning', 'painting', 'flooring']
        }
        
        # Check each activity for sequence violations
        for i, (activity, phase) in enumerate(zip(activity_texts, phases)):
            activity_lower = activity.lower()
            
            # Skip unknown phases
            if phase == 'unknown':
                continue
            
            # Get dependencies for this phase
            dependencies = critical_dependencies.get(phase, [])
            
            # Check if all dependencies are met
            for dep_phase in dependencies:
                if dep_phase in phase_positions:
                    # Check if current activity comes before its dependency
                    dep_positions = phase_positions[dep_phase]
                    if dep_positions and i < max(dep_positions):
                        # Found a violation
                        dep_activity_names = [activity_texts[pos] for pos in dep_positions]
                        issues[i] = {
                            'reason': f'{activity} must come after {dep_phase.replace("_", " ").title()} activities (found: {", ".join(dep_activity_names[-2:])})',
                            'severity': 'high',
                            'dependency': dep_phase
                        }
                        break
            
            # Additional specific checks based on activity keywords
            if not i in issues:  # Only check if no issue found yet
                
                # Permits must be first
                if phase == 'permits_approvals':
                    non_permit_before = [j for j in range(i) if phases[j] != 'permits_approvals']
                    if non_permit_before:
                        issues[i] = {
                            'reason': 'Permits and approvals must be obtained before any construction work begins',
                            'severity': 'critical'
                        }
                
                # HVAC/MEP finish should not be too early
                elif phase == 'mep_finish':
                    struct_phases = ['structural_steel', 'structural_concrete', 'masonry']
                    struct_exists = any(p in phase_positions for p in struct_phases)
                    if struct_exists:
                        struct_max = max([max(phase_positions.get(p, [-1])) for p in struct_phases])
                        if i < struct_max:
                            issues[i] = {
                                'reason': 'MEP finish work (HVAC, electrical, plumbing) must come after structural work is complete',
                                'severity': 'high'
                            }
                
                # Concrete pouring specific checks
                elif 'pour' in activity_lower or 'cast' in activity_lower or 'concrete place' in activity_lower:
                    # Check for formwork
                    if 'formwork' in phase_positions and phase_positions['formwork']:
                        if i < max(phase_positions['formwork']):
                            issues[i] = {
                                'reason': 'Concrete pouring requires formwork to be completed first',
                                'severity': 'high'
                            }
                    # Check for rebar
                    if 'rebar' in phase_positions and phase_positions['rebar']:
                        if i < max(phase_positions['rebar']):
                            issues[i] = {
                                'reason': 'Concrete pouring requires reinforcement (rebar) to be installed first',
                                'severity': 'high'
                            }
                
                # Waterproofing checks
                elif 'waterproof' in activity_lower:
                    if 'below' in activity_lower or 'foundation' in activity_lower:
                        if 'excavation' in phase_positions and phase_positions['excavation']:
                            if i < max(phase_positions['excavation']):
                                issues[i] = {
                                    'reason': 'Below-grade waterproofing must come after excavation is complete',
                                    'severity': 'high'
                                }
        
        return issues
    
    def _check_duplicates(self, schedule_df, activity_col):
        """Check for duplicate activity names in the schedule"""
        activity_names = schedule_df[activity_col].astype(str).tolist()
        
        # Find duplicates
        seen = {}
        duplicates_found = []
        
        for idx, name in enumerate(activity_names):
            if name in seen:
                if name not in [d['name'] for d in duplicates_found]:
                    duplicates_found.append({
                        'name': name,
                        'first_occurrence': seen[name] + 1,  # 1-based indexing
                        'duplicate_rows': [seen[name] + 1, idx + 1]
                    })
                else:
                    # Add to existing duplicate entry
                    for dup in duplicates_found:
                        if dup['name'] == name:
                            dup['duplicate_rows'].append(idx + 1)
                            break
            else:
                seen[name] = idx
        
        # Get activity IDs for duplicates if available
        activity_id_col = None
        for col in ['Activity ID', 'Activity_ID', 'Task ID', 'Task_ID']:
            if col in schedule_df.columns:
                activity_id_col = col
                break
        
        if activity_id_col and duplicates_found:
            for dup in duplicates_found:
                dup['activity_ids'] = [
                    str(schedule_df.iloc[row-1][activity_id_col]) 
                    for row in dup['duplicate_rows']
                ]
        
        return {
            'has_duplicates': len(duplicates_found) > 0,
            'duplicate_count': len(duplicates_found),
            'duplicates': duplicates_found,
            'total_activities': len(activity_names)
        }
    
    def _infer_phases(self, activity_texts):
        """Comprehensive phase inference from activity names"""
        phases = []
        for activity in activity_texts:
            activity_lower = activity.lower()
            
            # Permits and Approvals (Priority 1)
            if any(kw in activity_lower for kw in ['permit', 'approval', 'license', 'authorization', 'clearance']):
                phases.append('permits_approvals')
            
            # Site Preparation (Priority 2)
            elif any(kw in activity_lower for kw in ['site prep', 'site clear', 'grubbing', 'temporary fence', 'site access', 'mobilization']):
                phases.append('site_preparation')
            
            # Demolition (Priority 3)
            elif any(kw in activity_lower for kw in ['demolition', 'demolish', 'removal', 'strip out']):
                phases.append('demolition')
            
            # Surveying (Priority 4)
            elif any(kw in activity_lower for kw in ['survey', 'layout', 'marking', 'stake out', 'benchmark']):
                phases.append('surveying')
            
            # Excavation (Priority 5)
            elif any(kw in activity_lower for kw in ['excavat', 'earthwork', 'cut', 'dig', 'trenching', 'grading', 'earth moving']):
                phases.append('excavation')
            
            # Shoring and Support (Priority 6)
            elif any(kw in activity_lower for kw in ['shoring', 'sheet pil', 'bracing', 'underpinning', 'soil nail']):
                phases.append('shoring')
            
            # Dewatering (Priority 7)
            elif any(kw in activity_lower for kw in ['dewater', 'wellpoint', 'sump pump', 'water table']):
                phases.append('dewatering')
            
            # Piling (Priority 8)
            elif any(kw in activity_lower for kw in ['piling', 'pile', 'caisson', 'driven pile', 'bored pile', 'micropile']):
                phases.append('piling')
            
            # Blinding/Lean Concrete (Priority 9)
            elif any(kw in activity_lower for kw in ['blinding', 'lean concrete', 'mud mat', 'screed']):
                phases.append('blinding')
            
            # Waterproofing Below Grade (Priority 10)
            elif 'waterproof' in activity_lower and any(kw in activity_lower for kw in ['foundation', 'basement', 'below', 'membrane']):
                phases.append('waterproofing_below')
            
            # Formwork (Priority 11)
            elif any(kw in activity_lower for kw in ['formwork', 'shuttering', 'forms', 'falsework', 'centering']):
                phases.append('formwork')
            
            # Rebar/Reinforcement (Priority 12)
            elif any(kw in activity_lower for kw in ['rebar', 'reinforc', 'steel bar', 'mesh', 'stirrup', 'dowel']):
                phases.append('rebar')
            
            # Foundation/Concrete (Priority 13)
            elif any(kw in activity_lower for kw in ['foundation', 'footing', 'raft', 'mat foundation', 'pile cap']) or \
                 ('concrete' in activity_lower and any(kw in activity_lower for kw in ['pour', 'cast', 'place'])):
                phases.append('foundation')
            
            # Backfilling (Priority 14)
            elif any(kw in activity_lower for kw in ['backfill', 'compaction', 'fill', 'soil replacement']):
                phases.append('backfilling')
            
            # Structural Steel (Priority 15)
            elif any(kw in activity_lower for kw in ['steel erection', 'structural steel', 'steel frame', 'metal deck', 'steel beam', 'steel column']):
                phases.append('structural_steel')
            
            # Structural Concrete (Priority 16)
            elif any(kw in activity_lower for kw in ['slab', 'beam', 'column', 'core wall', 'shear wall']) and \
                 not any(kw in activity_lower for kw in ['steel', 'metal']):
                phases.append('structural_concrete')
            
            # Masonry (Priority 17)
            elif any(kw in activity_lower for kw in ['masonry', 'brick', 'block', 'cmu', 'stone work', 'mortar']):
                phases.append('masonry')
            
            # Roofing (Priority 18)
            elif any(kw in activity_lower for kw in ['roof', 'parapet', 'coping', 'flashing', 'membrane roof']):
                phases.append('roofing')
            
            # Waterproofing Above Grade (Priority 19)
            elif 'waterproof' in activity_lower and not any(kw in activity_lower for kw in ['foundation', 'basement', 'below']):
                phases.append('waterproofing_above')
            
            # Curtain Wall/Facade (Priority 20)
            elif any(kw in activity_lower for kw in ['curtain wall', 'facade', 'cladding', 'exterior panel', 'glazing', 'window', 'storefront']):
                phases.append('curtain_wall')
            
            # MEP Rough-in (Priority 21)
            elif any(kw in activity_lower for kw in ['rough-in', 'rough in', 'sleeve', 'conduit', 'ductwork', 'pipe rough']) or \
                 ('hvac' in activity_lower or 'mechanical' in activity_lower or 'electrical' in activity_lower or 'plumbing' in activity_lower) and \
                 'rough' in activity_lower:
                phases.append('mep_rough')
            
            # HVAC/Mechanical (Priority 22)
            elif any(kw in activity_lower for kw in ['hvac', 'ducting', 'ventilation', 'air handling', 'chiller', 'boiler', 'mechanical equip']):
                phases.append('mep_finish')
            
            # Electrical (Priority 23)
            elif any(kw in activity_lower for kw in ['electrical', 'wiring', 'cable', 'panel', 'switchgear', 'transformer', 'lighting']):
                phases.append('mep_finish')
            
            # Plumbing (Priority 24)
            elif any(kw in activity_lower for kw in ['plumbing', 'pipe', 'sanitary', 'storm drain', 'water supply', 'fixture']):
                phases.append('mep_finish')
            
            # Fire Protection (Priority 25)
            elif any(kw in activity_lower for kw in ['fire protection', 'sprinkler', 'fire alarm', 'fire pump']):
                phases.append('mep_finish')
            
            # Insulation (Priority 26)
            elif any(kw in activity_lower for kw in ['insulation', 'thermal', 'acoustic', 'fireproofing']):
                phases.append('insulation')
            
            # Drywall/Interior Walls (Priority 27)
            elif any(kw in activity_lower for kw in ['drywall', 'gypsum', 'partition', 'interior wall', 'framing']):
                phases.append('drywall')
            
            # Flooring (Priority 28)
            elif any(kw in activity_lower for kw in ['flooring', 'tile', 'carpet', 'vinyl', 'hardwood', 'epoxy floor']):
                phases.append('flooring')
            
            # Painting (Priority 29)
            elif any(kw in activity_lower for kw in ['paint', 'primer', 'coating', 'finish']):
                phases.append('painting')
            
            # Testing (Priority 30)
            elif any(kw in activity_lower for kw in ['test', 'inspection', 'check', 'verify']):
                phases.append('testing')
            
            # Commissioning (Priority 31)
            elif any(kw in activity_lower for kw in ['commission', 'startup', 'balancing', 'tab', 'handover']):
                phases.append('commissioning')
            
            # Landscaping (Priority 32)
            elif any(kw in activity_lower for kw in ['landscape', 'planting', 'irrigation', 'hardscape', 'paving']):
                phases.append('landscaping')
            
            # Final Activities (Priority 33)
            elif any(kw in activity_lower for kw in ['final inspection', 'punch list', 'certificate of occupancy', 'closeout']):
                phases.append('final_inspection')
            
            else:
                phases.append('unknown')
        
        return phases
    
    def _generate_correction(self, activity_name, phase, current_position):
        """Generate correction details with proper construction sequencing"""
        activity_lower = activity_name.lower()
        
        violation_type = 'sequence_anomaly'
        suggested_position = current_position
        
        # Check for specific violations based on activity type and phase
        if 'permit' in activity_lower:
            violation_type = 'permit_after_work'
            suggested_position = 1
        elif 'hvac' in activity_lower or 'ducting' in activity_lower:
            # HVAC should come after structural work
            violation_type = 'hvac_before_structure'
            suggested_position = max(current_position, 15)  # After structural work
        elif 'backfill' in activity_lower or 'compaction' in activity_lower:
            # Backfilling should come after foundation/concrete work
            violation_type = 'backfilling_before_foundation'
            suggested_position = max(current_position, 10)  # After foundation work
        elif 'formwork' in activity_lower or 'shuttering' in activity_lower:
            # Formwork needs excavation first
            if current_position < 3:
                violation_type = 'formwork_before_excavation'
                suggested_position = 4  # After excavation
        elif 'foundation' in activity_lower or 'concrete' in activity_lower:
            # Foundation/concrete needs excavation and formwork first
            if current_position < 5:
                violation_type = 'foundation_before_excavation'
                suggested_position = 6  # After formwork
        elif 'excavation' in activity_lower:
            # Excavation should be early in the sequence
            if current_position > 3:
                violation_type = 'sequence_anomaly'
                suggested_position = 2  # Early position
        elif 'testing' in activity_lower or 'commissioning' in activity_lower:
            violation_type = 'testing_before_installation'
            suggested_position = current_position + 2
        elif 'inspection' in activity_lower:
            violation_type = 'inspection_before_work'
            suggested_position = current_position + 1
        elif 'finishing' in activity_lower or 'painting' in activity_lower:
            violation_type = 'finishing_before_structure'
            suggested_position = max(current_position, 20)
        
        template = self.correction_templates.get(violation_type, self.correction_templates['sequence_anomaly'])
        
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
            
            // Use longer timeout for large files
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 300000); // 5 minute timeout
            
            fetch('/analyze', {
                method: 'POST',
                body: formData,
                signal: controller.signal
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    showError(data.error, data);
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
                    showError(data.error, data);
                } else {
                    displayResults(data);
                }
            })
            .catch(error => showError('Error: ' + error.message));
        }

        function showLoading(fileSize) {
            const resultsDiv = document.getElementById('results');
            resultsDiv.style.display = 'block';
            
            let message = 'Generating training data, training LSTM model, and analyzing your schedule...';
            if (fileSize && fileSize > 100) {
                message = `Processing large schedule (${fileSize} activities). This may take 1-3 minutes...`;
            }
            
            resultsDiv.innerHTML = `
                <div class="loading">
                    <h3>🧠 LSTM Neural Network Processing...</h3>
                    <p>${message}</p>
                    <div style="margin-top: 1rem;">
                        <div style="width: 200px; height: 4px; background: #e0e0e0; border-radius: 2px; margin: 0 auto;">
                            <div style="width: 50%; height: 100%; background: #4299e1; border-radius: 2px; animation: progress 2s ease-in-out infinite;"></div>
                        </div>
                    </div>
                </div>
                <style>
                    @keyframes progress {
                        0% { width: 0%; }
                        50% { width: 100%; }
                        100% { width: 0%; }
                    }
                </style>`;
        }

        function displayResults(data) {
            const resultsDiv = document.getElementById('results');
            resultsDiv.style.display = 'block';
            
            let html = '';
            
            // Duplicate warning
            if (data.analysis && data.analysis.duplicate_warning) {
                const dupWarn = data.analysis.duplicate_warning;
                html += `
                    <div class="error" style="background: #fff3cd; border-color: #ffc107; color: #856404;">
                        <h4>⚠️ Duplicate Activities Detected</h4>
                        <p>${dupWarn.message}</p>
                        <p><strong>Found ${dupWarn.total_duplicates} duplicate activity name(s):</strong></p>
                        <ul style="margin-top: 0.5rem;">`;
                
                dupWarn.duplicates.forEach(dup => {
                    html += `<li>"${dup.name}" appears ${dup.duplicate_rows.length} times (rows: ${dup.duplicate_rows.join(', ')})</li>`;
                });
                
                html += `</ul>
                        <p style="font-style: italic; margin-top: 0.5rem;">Activities have been automatically numbered (#2, #3, etc.) for analysis.</p>
                    </div>`;
            }
            
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
            
            // Check if results are truncated for large schedules
            if (data.analysis && data.analysis.is_truncated) {
                html += `
                    <div style="background: #e6f3ff; padding: 1rem 2rem; border: 1px solid #4299e1; margin: 0 2rem 1rem;">
                        <h4>📊 Large Schedule Notice</h4>
                        <p>Showing first ${data.analysis.displayed_count} of ${data.analysis.all_results_count} activities.</p>
                        <p>All ${data.analysis.all_results_count} activities were analyzed. See summary above for complete statistics.</p>
                    </div>`;
            }
            
            // Results rows
            if (data.analysis && data.analysis.results) {
                // Only show issues for very large schedules
                const resultsToShow = data.analysis.all_results_count > 200 
                    ? data.analysis.results.filter(r => !r.is_correct)
                    : data.analysis.results;
                
                if (data.analysis.all_results_count > 200 && resultsToShow.length < data.analysis.results.length) {
                    html += `
                        <div style="padding: 0.5rem 2rem; background: #fffacd;">
                            <p>ℹ️ Showing only activities with issues (${resultsToShow.length} of ${data.analysis.displayed_count})</p>
                        </div>`;
                }
                
                resultsToShow.forEach(result => {
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

        function showError(message, details) {
            const resultsDiv = document.getElementById('results');
            resultsDiv.style.display = 'block';
            
            let errorHtml = `<div class="error"><h3>❌ Error</h3><p>${message}</p>`;
            
            // If duplicate details are provided
            if (details && details.duplicate_details) {
                const dupInfo = details.duplicate_details;
                errorHtml += `
                    <div style="margin-top: 1rem;">
                        <h4>Duplicate Activity Names Found:</h4>
                        <p>Total Activities: ${dupInfo.total_activities} | Duplicates Found: ${dupInfo.duplicate_count}</p>
                        <ul style="margin-top: 0.5rem;">`;
                
                dupInfo.duplicates.forEach(dup => {
                    errorHtml += `<li><strong>"${dup.name}"</strong> appears in rows: ${dup.duplicate_rows.join(', ')}`;
                    if (dup.activity_ids) {
                        errorHtml += ` (IDs: ${dup.activity_ids.join(', ')})`;
                    }
                    errorHtml += `</li>`;
                });
                
                errorHtml += `
                        </ul>
                        <p style="margin-top: 1rem; font-style: italic;">
                            Please ensure all activity names are unique. Consider adding location identifiers, 
                            phase numbers, or other distinguishing details to make each activity name unique.
                        </p>
                    </div>`;
            }
            
            errorHtml += `</div>`;
            resultsDiv.innerHTML = errorHtml;
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
            # Special handling for duplicate errors
            if 'duplicate_details' in analysis_results:
                return jsonify({
                    'error': analysis_results['error'],
                    'message': analysis_results['message'],
                    'duplicate_details': analysis_results['duplicate_details']
                }), 400
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
    print("📊 CAPACITY:")
    print("   • Supports up to 1000 activities per schedule")
    print("   • Automatic batching for schedules > 100 activities")
    print("   • Handles duplicate activity names with warnings")
    print("=" * 60)
    
    # Increase timeout for large schedules
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    app.run(debug=True, host='0.0.0.0', port=8080, threaded=True)