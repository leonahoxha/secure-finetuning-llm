import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV log
log_df = pd.read_csv('out/base/training_log.csv')

plt.figure()
plt.plot(log_df['iter'], log_df['train_loss'], label='Train Loss')
plt.plot(log_df['iter'], log_df['val_loss'],   label='Validation Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training & Validation Loss over Iterations')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save to file
plt.savefig('out/base/base_curve.png', dpi=300)
print("Saved plot to out/base/base_curve.png")
