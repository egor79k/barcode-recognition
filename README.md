# barcode-recognition
### Запуск декодера с локализатором:
Аргументы:
- Изображение
- Локализатор: 1, 2
- Веса модели локализатора (здесь есть best21042023I.pt для локализатора 2)
- Дополнительно: выходное изображение
```
python3 decode.py <input image path> <localizer (1 or 2)> <localizer checkpoint path> [output image path]
```
Пример:
```
python3 decoder.py image.png 2 best21042023I.pt
```

### Подсчет статистики распознаваемости:
Аргументы:
- Файл разметки
- Декодер QR: opencv, zbar, zxing
- Декодер DM: libdmtx, zxing
- Локализатор: 1, 2
- Веса модели локализатора (здесь есть best21042023I.pt для локализатора 2)
```
python3 decode.py <markup file> <QR decoder> <DataMatrix decoder> <localizer> <localizer checkpoint>
```
Пример:
```
python3 decoder.py datast/markup.json opencv libdmtx 2 best21042023I.pt
```
