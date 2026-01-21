import matplotlib.pyplot as plt

def plot_accuracy_comparison():
    # ---------------------------------------------------------
    # DATA INPUT SECTION
    # Replace the values in the lists below with your actual results.
    # ---------------------------------------------------------

    # 1. Synchronous Implementation
    # Question 2 asks to run with 4, 8, and 16 cores.
    # In Sync, usually all cores act as workers (or 1 master + N-1 workers).
    # Adjust 'sync_workers' based on whether you count rank 0 as a worker.
    # Assuming Total Cores = Workers for Sync (common in all-reduce):
    sync_total_cores = [4, 8, 16]
    sync_accuracy = [87.88, 68.84, 68.67]  # <--- REPLACE with your Sync accuracies

    # 2. Asynchronous Implementation (2 Masters)
    # Question 5 asks to run 2 Masters with 4 and 8 TOTAL cores.
    # Workers = Total Cores - Masters
    # 4 cores - 2 masters = 2 workers
    # 8 cores - 2 masters = 6 workers
    async_2m_workers = [2, 6] 
    async_2m_accuracy = [92.3, 9.8]    # <--- REPLACE with your Async (2M) accuracies

    # 3. Asynchronous Implementation (4 Masters)
    # Question 5 asks to run 4 Masters with 8 and 16 TOTAL cores.
    # 8 cores - 4 masters = 4 workers
    # 16 cores - 4 masters = 12 workers
    async_4m_workers = [4, 12]
    async_4m_accuracy = [88.85, 8.92]    # <--- REPLACE with your Async (4M) accuracies

    # ---------------------------------------------------------
    # PLOTTING SECTION
    # ---------------------------------------------------------
    
    plt.figure(figsize=(10, 6))

    # Plot Synchronous
    plt.plot(sync_total_cores, sync_accuracy, marker='o', linestyle='-', 
             label='Synchronous', color='blue')

    # Plot Async (2 Masters)
    plt.plot(async_2m_workers, async_2m_accuracy, marker='s', linestyle='--', 
             label='Async (2 Masters)', color='green')

    # Plot Async (4 Masters)
    plt.plot(async_4m_workers, async_4m_accuracy, marker='^', linestyle='--', 
             label='Async (4 Masters)', color='red')

    # Graph formatting
    plt.title('Final Accuracy vs. Number of Workers', fontsize=14)
    plt.xlabel('Number of Workers', fontsize=12)
    plt.ylabel('Final Test Accuracy (%)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()

    # Save and Show
    output_filename = 'accuracy_vs_workers.png'
    plt.savefig(output_filename)
    print(f"Graph saved as {output_filename}")
    plt.show()

if __name__ == "__main__":
    plot_accuracy_comparison()
