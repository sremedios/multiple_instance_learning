from .utils import now

def write_log(log_file, host_id, cur_train_loss, cur_val_loss, cur_patience, best_loss, cur_epoch):
    update_log_file = False
    new_log_file = False

    with open(log_file, 'r') as f:
        logfile_data = [x.split() for x in f.readlines()]

        if (len(logfile_data) >= 1 and logfile_data[-1][1] != host_id)\
                or len(logfile_data) == 0:
            update_log_file=True
        if len(logfile_data) == 0:
            new_log_file=True

    if update_log_file:
        with open(log_file, 'a') as f:
            if new_log_file:
                f.write("{:<30}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\n".format(
                    "timestamp",
                    "host_id",
                    "cur_train_loss",
                    "cur_val_loss",
                    "cur_patience",
                    "best_loss",
                    "cur_epoch",
                    ))
            f.write("{:<30}\t{:<10}\t{:<10.4f}\t{:<10.4f}\t{:<10}\t{:<10.4f}\t{:<10}\n".format(
                now(),
                host_id,
                cur_train_loss,
                cur_val_loss,
                cur_patience,
                best_loss,
                cur_epoch,
                ))
