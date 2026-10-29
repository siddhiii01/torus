# Intern review task

Deadline: **30 October 2026, end of day.**

You are checking whether the engine read each page the way a person would. You are not changing the engine. Send one note. We decide what to fix.

## Setup

Clone this repository and stay on `main`. Do not create a branch. Do not install Python, PyTorch, or the OCR models. You will not run the engine.

```
git clone https://github.com/siddhiii01/local-ocr-engine.git
cd local-ocr-engine
git checkout main
git pull origin main
```

After the pull, this file is `REVIEW_FOR_INTERN.md` in the project root. Read it, then do the review below.

You will also be given a separate folder named `review`. It is not in git. It contains real customer documents. Do not commit it, do not upload it, and do not send it to anyone except the person who gave it to you.

## What to open

Inside the `review` folder, double-click `index.html`. It opens in a browser.

There are 10 pages. Do all 10, in this order:

1. `#03` page 1
2. `#34` pages 1, 2, 3, 4, 5, 6, and 7
3. `#06` page 1
4. `#37` page 1

Each page has a heading like `#03 invoice · page 1 · textlayer · ok`.

- `#03` is the file. `page 1` is which page inside that file.
- `textlayer` or `doctr` is which reader ran. Do not judge the reader. Judge the result.
- `ok` means the engine is willing to return the page. `needs_review` means it is not sure. A blank page should say `needs_review`.

Under the heading there are three parts:

- Left: the page, with a box around each word the engine found.
- Middle: those words in the order the engine read them.
- Right: a **Fields** table (`Field` and `Value`) and any **Tables**.

## What to check

On every page, do these four checks. About five to ten minutes for a full page.

**Boxes.** Look only at the picture. A box should sit on a word, not on empty paper and not halfway between two lines. A word you can read should have a box. A logo or a signature with no letters can be unboxed. Write `boxes pass` or `boxes miss`, and name the place if it is a miss.

**Order.** Read the middle list from the top, then look at the picture. The list should follow the page the way you would read it: top to bottom, and on a row, left to right across the whole row before the next row. Fail it if a column is read all the way down before the next column, if the title comes after the address, or if the list jumps from the bottom back to the top. Write `order pass` or `order miss`, and name the place.

**Words.** Pick five spots: the title, one name, one date, one long reference number, and the total. If the page has handwriting, include the handwriting. A wrong character, a missing word, two words glued together, or a word that is not on the page is a miss. Name the kind of spot and what is wrong. Do not paste a sentence from the document. Example: `word miss, handwritten date, the year digit is wrong`.

**Fields.** Look at the right-hand Field / Value table and any Tables. A field is a miss when the label and the value do not belong together, or the value is not what is printed next to that label on the page. A table is a miss when a cell is under the wrong column. Write `fields pass` or `fields miss`, and name the row. Example: `fields miss, date value is the invoice number`.

Also, once per page if you see them:

- Arabic or any non-English line: write `non-English not read`. Do not correct it.
- A ticked checkbox whose tick is not recorded: write `checkbox not recorded`.
- Stamp letters with no box, or only some of the letters read: write `stamp partial`.

`#37` is a blank page. Pass it only if the picture is empty, the lists are empty, and the heading says `needs_review`.

## What to report

Send one message by **30 October 2026, end of day**, to the person who assigned this. Use this shape for all 10 pages. `pass` means you looked and it matches. If you are unsure, write `unsure` and one sentence. Do not guess a pass.

```
#03 page 1
boxes: pass
order: miss, item table read down the columns
words: pass
fields: miss, date value is the invoice number
checkbox: none
stamp: none

#34 page 1
boxes: pass
order: pass
words: pass
fields: pass
checkbox: none
stamp: partial

#37 page 1
boxes: pass
order: pass
words: pass
fields: pass
blank: picture empty, lists empty, heading says needs_review
```

Do not paste sentences or numbers from the documents. Name the place and the kind of error.

## What you do not do

- Do not edit the HTML, the images, or any code.
- Do not run OCR.
- Do not open a pull request.
- Do not skip `#34` pages 2 to 7 because page 1 looked fine.
