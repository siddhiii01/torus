# How to review one OCR page

You are checking whether our engine read the page the way a person would. You are not fixing the engine. Send notes. We decide what to change.

The file to open is:

`D:\OCR-Engine\reports\review\index.html`

Double-click it. It opens in a browser. Do not copy this folder into git, email, or a chat outside the company. The pages are real customer documents.

## What is on the screen

Each block is one page. The heading looks like this:

`#03 invoice · page 1 · textlayer · ok`

- The number (`#03`) is the file.
- `page 1` is which page inside that file.
- `textlayer` or `doctr` is which reader ran. You do not judge the reader. You judge the result.
- `ok` means the engine is willing to return the page. `needs_review` means the engine is not sure. A blank page should say `needs_review`.

Under the heading:

- Left: the page photo. Red boxes are the words the engine found.
- Right: those same words, in the order the engine read them. Line 1 in the list is what the engine thinks comes first.

There are 10 pages. Do all 10. Order:

1. `#03` page 1
2. `#34` pages 1, 2, 3, 4, 5, 6, 7
3. `#06` page 1
4. `#37` page 1

## Three checks, in this order

Do them on every page. Spend about five to ten minutes on a full page, less on a short one.

### Check A. Boxes

Look only at the picture.

- A box should sit on a word, not on empty paper and not halfway between two lines.
- A word you can read should have a box. A logo or a signature with no letters can be unboxed.
- A stamp with letters: note it if the letters have no box.

Write `boxes pass` or `boxes miss`, plus where. Example: `boxes miss, bottom table, boxes sit on the line above the words`.

### Check B. Order

Cover the picture. Read the list on the right from the top. Then look back at the picture.

The list should follow the page the way you would read it: top to bottom, and on a row, left to right across the whole row before the next row.

Fail the page if:

- a column is read all the way down before the next column starts
- the title shows up after the address
- a total shows up before the lines it belongs to
- the list jumps from the bottom back to the top

Write `order pass` or `order miss`, plus where. Example: `order miss, item table is read down each column instead of across the row`.

### Check C. Words

Pick five spots, not the whole page: the title, one name, one date, one long reference number, and the total. If the page has handwriting, the five spots must include the handwriting.

Compare each spot to the list.

- Wrong character, missing word, two words glued together, or a word that is not on the page: that is a miss.
- Say what kind of spot it was, and what is wrong. Do not paste the whole page.
- Example: `word miss, handwritten date, the year digit is wrong`.
- Example: `word miss, total, the amount is missing`.
- A printed word that matches: do not write it out.

Arabic or any non-English line: write `non-English not read` once. Do not try to correct it. English is the scope.

A tick inside a checkbox: write `checkbox not recorded` if you can see a tick and the list only has the label, not which box is ticked.

## How to write the note

One block per page. Send the whole thing back in one message. Use this shape. Replace the examples.

```
#03 page 1
boxes: pass
order: miss, item table read down the columns
words: pass on title, date, total. miss on one description, last word dropped
checkbox: none
stamp: none

#34 page 1
boxes: pass
order: pass
words: pass
checkbox: none
stamp: partial, ring of the stamp has some letters and misses others

#37 page 1
boxes: pass
order: pass
words: pass
blank: picture is empty, list is empty, heading says needs_review
```

Rules for the note:

- `pass` means you looked, and it matches.
- A miss names the place (header, table, total, date, stamp, checkbox) and the kind of error (shifted box, wrong order, wrong character, missing, extra garbage word).
- Do not paste a sentence from the document.
- Do not say “looks fine” with no check name. We need the three words: boxes, order, words.
- If you are unsure, write `unsure` and one sentence about what you could not decide. Do not guess a pass.

## What you do not do

- Do not edit the HTML, the images, or any code.
- Do not run a new OCR.
- Do not send the pages or your notes to anyone except the person who gave you this file.
- Do not skip `#34` pages 2 to 7 because page 1 looked fine. Each page is its own document.

When all 10 blocks are written, send them. That is the whole task.
