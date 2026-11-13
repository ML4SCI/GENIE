import os
import pickle
import logging
import awkward as ak
import numpy as np
import dask
import gudhi as gd
from scnn.build_matrices import build_boundaries, build_laplacians
import dask
from scnn.chebyshev import normalize, assemble
from scnn.scnn import coo2tensor
from dask import delayed, compute
import torch
from dask.distributed import Client
from scipy.sparse import csr_matrix
from tqdm import tqdm


class EventProcessor:
    def __init__(self, pkl_file, columns, label_column=None):
        """
        Initialize the EventProcessor class.

        Parameters:
        - pkl_file (str): Path to the .pkl file containing event data.
        - columns (list of str): List of column names to extract from each event's data.
        - label_column (str, optional): The name of the column to extract as labels.
        """
        self.pkl_file = pkl_file
        self.columns = columns
        self.label_column = label_column
        self.event_dicts = self._load_pkl_file()

    def _load_pkl_file(self):
        """Load the event dictionary from the provided pickle file."""
        with open(self.pkl_file, 'rb') as f:
            return pickle.load(f)

    def _extract_coordinates(self):
        """
        Extracts specified columns from the event_dicts.

        Returns:
        - coordinates (dict): A dictionary with event IDs as keys and stacked arrays of specified columns as values.
        - labels (dict): A dictionary with event IDs as keys and label column values, if label_column is provided.
        """
        coordinates = {}
        labels = {}

        for event_id, data in self.event_dicts.items():
            extracted_columns = []
            for column in self.columns:
                if column in data:
                    if column == 'phi':
                        phi = ak.to_numpy(data['phi'])
                        extracted_columns.append(np.sin(phi))
                        extracted_columns.append(np.cos(phi))
                    else:
                        extracted_columns.append(ak.to_numpy(data[column]))

            # Stack the extracted columns along the last axis
            if extracted_columns:
                coordinates[event_id] = np.stack(extracted_columns, axis=-1)

            # Extract the label column if provided
            if self.label_column and self.label_column in data:
                labels[event_id] = ak.to_numpy(data[self.label_column])

        return coordinates, labels

    def _compute_rips(self, event_id, coord, max_dimension, sparsity, filtration_val):
        """
        Helper function to compute Vietoris-Rips complex for a single event.

        Parameters:
        - event_id (str): The event ID.
        - coord (numpy.ndarray): The coordinates of the event.
        - max_dimension (int): The maximum dimension of the simplicial complex.

        Returns:
        - tuple: (laplacians, boundaries) for the event.
        """

        def extract_simplices(simplex_tree):
            simplices = [dict() for _ in range(simplex_tree.dimension() + 1)]
            for simplex, _ in simplex_tree.get_skeleton(simplex_tree.dimension()):
                k = len(simplex)
                simplices[k - 1][frozenset(simplex)] = len(simplices[k - 1])
            return simplices

        coord = (coord - coord.mean(axis=0)) / (coord.std(axis=0) + 1e-8)
        rips_complex = gd.RipsComplex(points=coord, sparse=sparsity)
        st = rips_complex.create_simplex_tree(max_dimension=max_dimension)
        _ = st.prune_above_filtration(filtration_val)
        assert st.num_vertices() == coord.shape[0]
        # assert st.dimension() == max_dimension

        simplices = extract_simplices(st)
        boundaries = build_boundaries(simplices)
        laplacians = build_laplacians(boundaries)

        return laplacians, boundaries, simplices

    def _create_dynamic_features(self, num_nodes, num_edges, num_faces, feature_dim=4):
        """
        Create feature tensors with dynamic sizes based on actual simplex counts.
        
        Parameters:
        - num_nodes (int): Actual number of nodes in this event
        - num_edges (int): Actual number of edges in this event
        - num_faces (int): Actual number of faces in this event
        - feature_dim (int): Number of features per node/edge/face
        
        Returns:
        - list of tensors: Dynamic feature tensors for nodes, edges, faces
        """
        xs_temp = [
            torch.rand((feature_dim, num_nodes)),  # Degree 0 input (nodes)
            torch.rand((feature_dim, num_edges)),  # Degree 1 input (edges)
            torch.rand((feature_dim, num_faces))   # Degree 2 input (faces)
        ]
        return xs_temp

    def _process_event_data(self, event_id, coord, max_dimension, sparsity, filtration_val, idx, total_events):
        """
        Process a single event and return all computed data.
        Handles both 2D and 1D cases efficiently.
        
        Returns:
        - dict or None: Dictionary with laplacians, boundaries, features, label, simplices if successful, None otherwise
        """
        logging.info(f'Processing event {event_id} ({idx + 1}/{total_events})')
        
        try:
            # Compute the laplacians and boundaries for the event
            laplacians, boundaries, simplices = self._compute_rips(
                event_id, coord, max_dimension, sparsity, filtration_val
            )

            num_nodes = laplacians[0].shape[0]
            
            # Handle 2D simplicial complex (nodes, edges, faces)
            if len(boundaries) == 2:
                topdim = 2
                num_edges = boundaries[0].shape[1]
                num_faces = boundaries[1].shape[1]

                # Create dynamic feature tensors
                xs_temp = self._create_dynamic_features(num_nodes, num_edges, num_faces)
                
                # Convert to tensors
                Ls = [coo2tensor(normalize(laplacians[k])) for k in range(topdim + 1)]
                Ds = [coo2tensor(boundaries[k].transpose()) for k in range(topdim)]
                adDs = [coo2tensor(boundaries[k]) for k in range(topdim)]

            # Handle 1D simplicial complex (nodes, edges only)
            elif len(boundaries) == 1:
                topdim = 1
                num_edges = boundaries[0].shape[1]
                num_faces = 1  # No faces exist in the chosen filtration

                # Create dynamic feature tensors with placeholder for faces
                xs_temp = [
                    torch.rand((4, num_nodes)),
                    torch.rand((4, num_edges)),
                    torch.zeros((4, num_faces))  # Placeholder for missing faces
                ]
                
                # Convert to tensors
                Ls = [coo2tensor(normalize(laplacians[k])) for k in range(topdim + 1)]
                Ds = [coo2tensor(boundaries[k].transpose()) for k in range(topdim)]
                adDs = [coo2tensor(boundaries[k]) for k in range(topdim)]
            
            else:
                logging.warning(f"Event {event_id} has unexpected boundary structure, skipping")
                return None

            return {
                'laplacians': laplacians,
                'boundaries': boundaries,
                'node_feats': xs_temp,
                'simplices': simplices
            }
            
        except Exception as e:
            logging.warning(f"Error while processing event {event_id} at index {idx}: {e}")
            return None

    def compute_lapl_and_bounds(self, step, limit, max_dimension=2, sparsity=0.3, filtration_val=1.0,
                                output_dir="scnn/bounds_and_laps", entity="train"):
        """
        OPTIMIZED: Compute laplacians and boundaries for events in chunks with improved efficiency.
        
        Key improvements:
        1. Dynamic feature handling - adapts to actual number of simplices
        2. Better memory management with chunked processing
        3. Reduced code duplication with helper methods
        4. Progress tracking with tqdm
        5. More robust error handling

        Parameters:
        - step (int): Number of events to process at a time.
        - limit (int, optional): Maximum number of events to process (None for no limit).
        - max_dimension (int, optional): The maximum dimension of the simplicial complexes (default is 2).
        - sparsity (float, optional): Sparsity factor for Rips complex construction.
        - filtration_val (float, optional): Filtration value for pruning.
        - output_dir (str, optional): Directory to store the laplacians and boundaries (default is 'scnn/bounds_and_laps').
        - entity (str, optional): Prefix for output files (default is 'train').

        Returns:
        - None
        """
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Created directory: {output_dir}")

        # Extract coordinates
        coordinates, labels = self._extract_coordinates()
        total_events = len(coordinates)
        logging.info(f'Total events: {total_events}')

        # Apply the limit if specified
        if limit is not None:
            total_events = min(limit, total_events)
            logging.info(f'Limiting to the first {total_events} events')

        chunk_count = 0
        all_laplacians, all_boundaries, all_node_feats, all_labels, all_simpl = [], [], [], [], []

        # Process events with progress bar
        event_items = list(coordinates.items())
        for idx in tqdm(range(total_events), desc=f"Processing {entity} events"):
            event_id, coord = event_items[idx]
            
            # Process single event
            result = self._process_event_data(
                event_id, coord, max_dimension, sparsity, filtration_val, idx, total_events
            )
            
            # Skip if processing failed
            if result is None:
                continue
            
            # Append the results
            all_laplacians.append(result['laplacians'])
            all_boundaries.append(result['boundaries'])
            all_node_feats.append(result['node_feats'])
            all_labels.append(labels[event_id])
            all_simpl.append(result['simplices'])

            # Save chunk if step size is reached
            if (idx + 1) % step == 0 or (idx + 1) == total_events:
                chunk_count += 1
                logging.info(f'Saving chunk {chunk_count} with {len(all_laplacians)} events')

                # Save results in chunks
                np.savez_compressed(os.path.join(output_dir, f'{entity}_laplacians_{chunk_count}.npz'), *all_laplacians)
                np.savez_compressed(os.path.join(output_dir, f'{entity}_boundaries_{chunk_count}.npz'), *all_boundaries)
                np.savez_compressed(os.path.join(output_dir, f'{entity}_node_feats_{chunk_count}.npz'), *all_node_feats)
                np.savez_compressed(os.path.join(output_dir, f'{entity}_labels_{chunk_count}.npz'), *all_labels)
                np.savez_compressed(os.path.join(output_dir, f'{entity}_simpl_{chunk_count}.npz'), *all_simpl)

                # Clear lists to free memory
                all_laplacians, all_boundaries, all_node_feats, all_labels, all_simpl = [], [], [], [], []

        logging.info('All chunks saved successfully.')


if __name__ == "__main__":
    pkl_path = os.path.join(os.getcwd(), 'downloads/processed/')
    pkl_file = os.path.join(pkl_path, 'train_data.pkl')
    columns_to_extract = ['pt', 'y', 'phi']  # or ['eta', 'phi'], etc.
    label_column = 'label'  # Specify the label column if needed

    # Initialize the processor
    processor = EventProcessor(pkl_file, columns_to_extract, label_column)

    processor.compute_lapl_and_bounds(step=20000, limit=None, max_dimension=2, sparsity=0.8, filtration_val=np.inf, entity="train")
    # Use filtration value around 10-15
    # FIXED: Added handling for dynamic simplices with _create_dynamic_features() and _process_event_data()
    # Feature vectors now adapt to actual number of nodes/edges/faces per event

    del processor, pkl_file

    pkl_file = os.path.join(pkl_path, 'test_data.pkl')

    # Initialize the processor
    processor = EventProcessor(pkl_file, columns_to_extract, label_column)

    processor.compute_lapl_and_bounds(step=10000, limit=None, max_dimension=2, sparsity=0.8, filtration_val=np.inf, entity="test")

    del processor, pkl_file

    pkl_file = os.path.join(pkl_path, 'val_data.pkl')

    # Initialize the processor
    processor = EventProcessor(pkl_file, columns_to_extract, label_column)

    processor.compute_lapl_and_bounds(step=100, limit=2000, max_dimension=2, sparsity=0.8, filtration_val=np.inf, entity="val")

    del processor, pkl_file