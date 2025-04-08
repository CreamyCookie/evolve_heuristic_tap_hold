# Evolving heuristic tap hold functions
This code uses the [jenetics library](https://github.com/jenetics/jenetics) to find three functions that can be used in a [heuristic tap hold](https://github.com/CreamyCookie/qmk_userspace/tree/main/keyboards/ducktopus/keymaps/vial/features) setup.

* One tries to predict the duration overlap above which a key combination is considered a hold instead of a tap. Example: <kbd>LCTL_T(KC_A)</kbd> down, <kbd>KC_V</kbd> down (this one always runs)
* One estimates hold or tap for a wrapped case. Example: <kbd>LCTL_T(KC_A)</kbd> down, <kbd>KC_V</kbd> down, <kbd>KC_V</kbd> up (`V` is wrapped)
* One does the same for a triple-down case. Example: <kbd>LCTL_T(KC_A)</kbd> down, <kbd>KC_V</kbd> down, <kbd>KC_E</kbd> down

They do so by using data points that are available at the time of the respective event. The duration between the tap hold key and the next key, is a data point in all three, but the duration between `V` and `E` in the triple-down case is obviously only available in such a situation.

This program uses the training data created via the [analyze_convert_keystroke_data](https://github.com/CreamyCookie/analyze_convert_keystroke_data) script.

The functions that were found so far are in [evolve_tap_hold_functions.kt](https://github.com/CreamyCookie/evolve_heuristic_tap_hold/blob/main/src/main/kotlin/evolve_tap_hold_functions.kt) in the `initialPopulation` variable.

# How to use
Download [training_data.csv.gz](https://github.com/CreamyCookie/analyze_convert_keystroke_data/tree/main/dataset).

Move it into this repository.

Configure via the [settings.kt](src/main/kotlin/settings.kt).

Start via the [Main.kt](src/main/kotlin/Main.kt).