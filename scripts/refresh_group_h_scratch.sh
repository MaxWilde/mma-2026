#!/bin/bash
#
# Emergency scratch-retention refresh for all CASTLE group_h scratch data.
#
# Submit a dry run:
#   sbatch scripts/refresh_group_h_scratch.sh
#
# Apply the access-time refresh:
#   sbatch --export=ALL,APPLY=1 scripts/refresh_group_h_scratch.sh
#
# The default target is the complete group_h shared-scratch directory:
#   /scratch-shared/group_h
#
# This currently includes:
#   data_goncalo/
#   data_goncalo_dashboard_overlay/
#   models/
#
# This updates access time (atime) only. It does not alter file contents or
# modification time (mtime). Scratch remains temporary, unbacked storage; this
# job cannot guarantee retention and is not a substitute for project/archive
# storage.

#SBATCH --job-name=refresh-group-h-scratch
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=/home/scur0260/scratch-refresh-%j.log

set -uo pipefail

ROOT="${ROOT:-/scratch-shared/group_h}"
APPLY="${APPLY:-0}"
NOW_EPOCH="$(date +%s)"
WARNING_AGE_DAYS="${WARNING_AGE_DAYS:-12}"

case "$ROOT" in
    /scratch-shared/group_h|\
    /gpfs/scratch1/shared/group_h)
        ;;
    *)
        echo "Refusing unexpected ROOT: $ROOT" >&2
        echo "Allowed roots are the canonical group_h shared-scratch paths." >&2
        exit 2
        ;;
esac

if [[ ! -d "$ROOT" ]]; then
    echo "Scratch root does not exist: $ROOT" >&2
    exit 2
fi

echo "CASTLE group_h scratch refresh"
echo "Started: $(date --iso-8601=seconds)"
echo "Host: $(hostname)"
echo "Root: $ROOT"
echo "Mode: $([[ "$APPLY" == "1" ]] && echo APPLY || echo DRY-RUN)"
echo

echo "Scanning current regular-file access ages..."
find "$ROOT" -xdev -type f -printf '%A@\n' 2>/dev/null |
awk -v now="$NOW_EPOCH" -v warning="$WARNING_AGE_DAYS" '
    {
        age_days = (now - $1) / 86400
        files++
        if (age_days >= warning) warning_count++
        if (age_days >= 14) expired_count++
        if (age_days > oldest) oldest = age_days
    }
    END {
        printf "Regular files: %d\n", files
        printf "Access age >= %.0f days: %d\n", warning, warning_count
        printf "Access age >= 14 days: %d\n", expired_count
        printf "Oldest access age: %.1f days\n", oldest
    }
'

echo
echo "Breakdown by top-level group_h directory:"
find "$ROOT" -xdev -mindepth 2 -type f -printf '%P\t%A@\n' 2>/dev/null |
awk -F '\t' -v now="$NOW_EPOCH" -v warning="$WARNING_AGE_DAYS" '
    {
        split($1, path_parts, "/")
        top = path_parts[1]
        age_days = (now - $2) / 86400
        files[top]++
        if (age_days >= warning) warning_count[top]++
        if (age_days >= 14) expired_count[top]++
        if (age_days > oldest[top]) oldest[top] = age_days
    }
    END {
        printf "%-36s %12s %12s %12s %12s\n",
               "Directory", "Files", ">= warning", ">= 14 days", "Oldest days"
        for (top in files) {
            printf "%-36s %12d %12d %12d %12.1f\n",
                   top, files[top], warning_count[top], expired_count[top], oldest[top]
        }
    }
'

if [[ "$APPLY" != "1" ]]; then
    echo
    echo "Dry run only; no timestamps were changed."
    echo "To apply:"
    echo "  cd /home/scur0260/mma-2026"
    echo "  sbatch --export=ALL,APPLY=1 scripts/refresh_group_h_scratch.sh"
    exit 0
fi

echo
echo "Refreshing access times for regular files..."
if ! find "$ROOT" -xdev -type f -exec touch -a -c -- {} +; then
    echo "ERROR: one or more files could not be refreshed." >&2
    exit 1
fi

echo "Refreshing access times for directories..."
if ! find "$ROOT" -xdev -depth -type d -exec touch -a -c -- {} +; then
    echo "ERROR: one or more directories could not be refreshed." >&2
    exit 1
fi

echo
echo "Verifying refreshed regular-file access ages..."
FAILURES="$(
    find "$ROOT" -xdev -type f -printf '%A@ %p\n' 2>/dev/null |
    awk -v now="$(date +%s)" '
        (now - $1) > 3600 {
            failures++
            if (failures <= 20) {
                $1 = ""
                sub(/^ /, "")
                print
            }
        }
        END {
            if (failures > 20) {
                printf "... and %d more\n", failures - 20
            }
        }
    '
)"

if [[ -n "$FAILURES" ]]; then
    echo "ERROR: files still showing access times older than one hour:" >&2
    echo "$FAILURES" >&2
    exit 1
fi

echo "Refresh completed successfully: $(date --iso-8601=seconds)"
echo
echo "Reminder: /scratch-shared has no backup and no guaranteed minimum"
echo "retention. Move irreplaceable data to project or archive storage."
