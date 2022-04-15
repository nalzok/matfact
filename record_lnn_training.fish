#!/usr/bin/env fish
for GROUND_TRUTH in reflector rank_one just_random
    for HIDDEN in 4 8 16
        echo "GROUND_TRUTH=$GROUND_TRUTH, HIDDEN=$HIDDEN"
        asciinema rec \
            --command="pipenv run python3 -m lnn.cli \
                --ground_truth $GROUND_TRUTH --p 8 \
                --features 8 --features $HIDDEN --features 8 \
                --epochs 16384" \
            --title "Linear Neural Network Training [ground_truth=$GROUND_TRUTH, features=(8, $HIDDEN, 8)]" \
            --yes
    end
end
