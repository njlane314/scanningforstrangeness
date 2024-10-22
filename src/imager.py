import os
import csv
import argparse
import numpy as np

def create_histograms(x, z, adc, flags, x_bounds, z_bounds, image_size, view):
    print(f"Creating histograms for view {view}.")
    print(f"x_bounds: {x_bounds}, z_bounds: {z_bounds}, image_size: {image_size}")
    
    if len(x) != len(z) or len(x) != len(adc):
        print(f"Error: Mismatched lengths: x({len(x)}), z({len(z)}), adc({len(adc)})")
        return None, None
    
    image_height, image_width = image_size
    x_min, x_max = x_bounds
    z_min, z_max = z_bounds

    z_bins = np.linspace(z_min - 0.5, z_max + 0.5, image_height + 1)
    x_bins = np.linspace(x_min - 0.5, x_max + 0.5, image_width + 1)

    print(f"Bins created: x_bins = {len(x_bins)}, z_bins = {len(z_bins)}")

    input_histogram, _, _ = np.histogram2d(z, x, bins=[z_bins, x_bins], weights=adc)
    input_histogram = input_histogram.astype(float)

    target_histogram = np.zeros_like(input_histogram, dtype='float')

    for i in range(len(x)):
        x_idx = np.clip(np.digitize(x[i], x_bins) - 1, 0, image_width - 1)
        z_idx = np.clip(np.digitize(z[i], z_bins) - 1, 0, image_height - 1)

        particle_class = 0
        for flag_index, flag in enumerate(flags[i]):
            if flag:
                particle_class = flag_index + 1

        target_histogram[z_idx, x_idx] = particle_class

    print(f"Histograms created for event.")
    return input_histogram, target_histogram


def parse_data(data):
    try:
        print(f"Parsing data: {data[:10]}...") 
        if data[-1] == '1':
            data = data[:-1] 

        n_hits = int(data[0])
        n_flags = int(data[1])

        print(f"n_hits: {n_hits}, n_flags: {n_flags}")

        x_vtx = float(data[2])
        z_vtx = float(data[3])

        evt_drift_min, evt_drift_max = float(data[4]), float(data[5])
        evt_wire_min, evt_wire_max = float(data[6]), float(data[7])
        intr_drift_min, intr_drift_max = float(data[8]), float(data[9])
        intr_wire_min, intr_wire_max = float(data[10]), float(data[11])

        print(f"Event vertex: x = {x_vtx}, z = {z_vtx}")
        print(f"Event extents: drift({evt_drift_min}, {evt_drift_max}), wire({evt_wire_min}, {evt_wire_max})")
        print(f"Interaction extents: drift({intr_drift_min}, {intr_drift_max}), wire({intr_wire_min}, {intr_wire_max})")

        hit_data_start_index = 12
        hit_data = data[hit_data_start_index:]

        expected_length = n_hits * (3 + n_flags)
        print(f"Expected hit data length: {expected_length}, Actual length: {len(hit_data)}")

        if len(hit_data) < expected_length:
            raise ValueError("Inconsistent hit data length")

        hit_x = np.array(hit_data[0::(3 + n_flags)], dtype=float)
        hit_z = np.array(hit_data[1::(3 + n_flags)], dtype=float)
        hit_adc = np.array(hit_data[2::(3 + n_flags)], dtype=float)

        print(f"Hit x sample: {hit_x[:5]}, z sample: {hit_z[:5]}, adc sample: {hit_adc[:5]}") 

        hit_flags = []
        for i in range(n_flags):
            hit_flags.append(np.array(hit_data[(3 + i)::(3 + n_flags)], dtype=float))
            print(f"Flag {i} data sample: {hit_flags[i][:5]}") 

        return (x_vtx, z_vtx, evt_drift_min, evt_drift_max, evt_wire_min, evt_wire_max,
                intr_drift_min, intr_drift_max, intr_wire_min, intr_wire_max), (hit_x, hit_z, hit_adc, np.vstack(hit_flags).T)

    except (ValueError, IndexError) as e:
        print(f"Error parsing event data: {e}")
        return None, None


def process_event(data, output_folder, event_number, view, image_size):
    print(f"Processing event {event_number}, view {view}.")
    metadata, hit_data = parse_data(data)
    if metadata is None or hit_data is None:
        print(f"Skipping event {event_number} due to data parsing issues.")
        return

    print(f"Parsed metadata and hit data for event {event_number}.")

    (x_vtx, z_vtx, evt_drift_min, evt_drift_max, evt_wire_min, evt_wire_max,
     intr_drift_min, intr_drift_max, intr_wire_min, intr_wire_max), (hit_x, hit_z, hit_adc, hit_flags) = metadata, hit_data

    input_histogram, target_histogram = create_histograms(
        hit_x, hit_z, hit_adc, hit_flags,
        (intr_drift_min, intr_drift_max), (intr_wire_min, intr_wire_max),
        image_size, view
    )

    input_output_folder = os.path.join(output_folder, "input")
    target_output_folder = os.path.join(output_folder, "target")
    os.makedirs(input_output_folder, exist_ok=True)
    os.makedirs(target_output_folder, exist_ok=True)

    input_filename = os.path.join(input_output_folder, f"image_{event_number}.npz")
    target_filename = os.path.join(target_output_folder, f"image_{event_number}.npz")

    np.savez_compressed(input_filename, input_histogram)
    np.savez_compressed(target_filename, target_histogram)

    print(f"Event {event_number}: Saved histograms to {input_filename} and {target_filename}")


def process_file(input_file, output_folder, view, image_size, num_events=None):
    print(f"Processing file: {input_file}")
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        for event_number, row in enumerate(reader):
            if num_events is not None and event_number >= num_events:
                break 
            print(f"Processing event {event_number + 1}")
            process_event(row, output_folder, event_number, view, image_size)

            if (event_number + 1) % 10 == 0:
                print(f"Processed {event_number + 1} events.")

        print(f"Completed processing {event_number + 1} events.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--raw_dir', type=str, required=True)
    parser.add_argument('-p', '--processed_dir', type=str, required=True)
    parser.add_argument('-f', '--file_prefix', type=str, default="training_output_")
    parser.add_argument('-s', '--image_size', type=int, nargs=2, default=[256, 256])
    parser.add_argument('-n', '--num_events', type=int, default=None)

    args = parser.parse_args()

    for view in ["U", "V", "W"]:
        input_file = os.path.join(args.raw_dir, f"{args.file_prefix}{view}.csv")
        output_folder = os.path.join(args.processed_dir, f"images_{view}")
        os.makedirs(output_folder, exist_ok=True)

        print(f"Starting processing for view {view}.")
        process_file(input_file, output_folder, view, image_size=tuple(args.image_size), num_events=args.num_events)
