import matplotlib.pyplot as plt

# Define your data
classes = ["Naive", "Theoretical\nAutoSearch", "Practical\nAutoSearch", "Practical\nAutoSearch\n& CudaGraph"]
durations = [70, 63, 71, 70]  # durations in milliseconds
errors = [1, 1, 1, 1]  # error for each bar (±1 ms)

# Create bar chart with error bars
plt.bar(classes, durations, yerr=errors, capsize=5, color="skyblue", edgecolor="black")

# Add labels and title
plt.xlabel("Implementation")
plt.ylabel("Duration Time (ms)")
# plt.title("Duration by Implementation")

plt.tight_layout()             # auto-adjust everything

# Save the plot
plt.savefig("single_gpu_duration_by_implementation.pdf")