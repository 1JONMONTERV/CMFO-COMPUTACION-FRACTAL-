import glob
import os
import sys

# Try imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import pandas as pd
except ImportError as e:
    print("Error: Missing Refdependencies.")
    print(f"Details: {e}")
    print("Please run: pip install matplotlib pandas")
    sys.exit(1)

def main():
    print("=== CMFO Soliton Visualizer ===")

    # Find files
    # Look in current dir and build dir just in case
    files = sorted(glob.glob("soliton_step_*.csv"))
    if not files:
        files = sorted(glob.glob("build/soliton_step_*.csv"))
        
    if not files:
        print("Error: No 'soliton_step_*.csv' files found.")
        print("Run the C simulation './test_soliton' first.")
        return

    print(f"Found {len(files)} frames. Generating GIF...")

    # Setup Plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    line, = ax.plot([], [], 'b-', lw=2, label='Phi Field')
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    # Simulation params from C code (hardcoded for visualization consistency)
    ax.set_xlim(-20, 20) 
    ax.set_ylim(-1.0, 7.5) # Roughly -1 to 2pi+1
    ax.set_xlabel('Position (x)')
    ax.set_ylabel('Field (Phi)')
    ax.set_title('CMFO Sine-Gordon Collision (Kink-Antikink)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    def update(frame_file):
        try:
            df = pd.read_csv(frame_file)
            x = df['x']
            phi = df['phi']
            line.set_data(x, phi)
            
            # Extract step number from filename
            # filename format: ...soliton_step_XXXX.csv
            base = os.path.basename(frame_file)
            step = base.split('_')[-1].split('.')[0]
            time_text.set_text(f'Step: {step}')
            return line, time_text
        except Exception as e:
            print(f"Error reading {frame_file}: {e}")
            return line, time_text

    ani = animation.FuncAnimation(fig, update, frames=files,
                                  init_func=init, blit=True)

    output_file = "soliton_collision.gif"
    try:
        print(f"Saving to {output_file} (this may take a moment)...")
        ani.save(output_file, writer='pillow', fps=10)
        print(f"[SUCCESS] Animation saved to {output_file}")
    except Exception as e:
        print(f"[FAIL] Could not save GIF: {e}")
        # Try plotting just the last frame as fallback
        print("Saving last frame as png instead...")
        update(files[-1])
        plt.savefig("soliton_final.png")
        print("Saved soliton_final.png")

if __name__ == "__main__":
    main()
