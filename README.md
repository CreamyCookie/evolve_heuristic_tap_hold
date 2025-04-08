# Evolving heuristic tap hold functions
This code uses the [jenetics library](https://github.com/jenetics/jenetics) to find three functions that can be used in a [heuristic tap hold](https://github.com/CreamyCookie/qmk_userspace/keyboards/ducktopus/keymaps/vial/features/) setup.

* One tries to predict the duration overlap above which a key combination is considered a hold instead of a tap.
* One guesses if a wrapped (<kbd>LCTL_T(KC_A)</kbd> down, <kbd>KC_V</kbd> down, <kbd>KC_V</kbd> up, ..) is hold or tap.
* One does the same for a triple-down situation ((<kbd>LCTL_T(KC_A)</kbd> down, <kbd>KC_V</kbd> down, <kbd>KC_E</kbd> down, )

They do so using data points such as that are available at the time of the event. One example data point is the duration between the tap hold key and the next key.

This program uses the training data created via the [analyze_convert_keystroke_data](https://github.com/CreamyCookie/analyze_convert_keystroke_data) script.

# How to use
Download [training_data.csv.gz](https://github.com/CreamyCookie/analyze_convert_keystroke_data/tree/main/dataset).

Move it into this repository.

Configure via the [settings.kt](src/main/kotlin/settings.kt).

Start via the [Main.kt](src/main/kotlin/Main.kt).