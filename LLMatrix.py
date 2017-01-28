class MatrixIndexError(Exception):
    '''An attempt has been made to access an invalid index in this matrix'''


class MatrixDimensionError(Exception):
    '''An attempt has been made to perform an operation on this matrix which
    is not valid given its dimensions'''


class MatrixInvalidOperationError(Exception):
    '''An attempt was made to perform an operation on this matrix which is
    not valid given its type'''


class MatrixNode(): 
    def __init__(self, contents, right=None, down=None, row=None, col=None):
        '''(MatrixNode, obj, MatrixNode, MatrixNode) -> NoneType
            Create a new node holding contents, that is linked to right
            and down in a matrix
        '''
        self._contents = contents
        self._right = right
        self._down = down
        self._row = row
        self._col = col

    def __str__(self):
        '''(MatrixNode) -> str
            Return the string representation of this node
        '''
        return str(self._contents)

    def get_row(self):
        return self._row

    def set_row(self, row):
        self._row = row

    def get_col(self):
        return self._col

    def set_col(self, col):
        self._col = col

    def _add_right(self, new_node, col_index):
        ''' (MatrixNode, MatrixNode, int) -> NoneType
            travel right through the list and add a node at col_index
        '''
        if(self.get_right()): 
            if(self._col < col_index):
                while(self.get_right() is not None and
                      self.get_right().get_col() < col_index):
                    self = self.get_right()
                new_node.set_right(self.get_right())
                self.set_right(new_node)
        else:
            # the next element is None so overwrite it with new_node
            self.set_right(new_node)

    def _add_down(self, new_node, row_index):
        ''' (MatrixNode, MatrixNode, int) -> NoneType
            Travel down through the list and add a node at row_index
        '''
        if(self.get_down()):
            # we can't attach something behind the node
            if(self._row < row_index):
                while(self.get_down() is not None and
                      self.get_down().get_row() < row_index):
                    self = self.get_down()
                new_node.set_down(self.get_down())
                self.set_down(new_node)
        else:
            # the next element is None so overwrite it with new_node
            self.set_down(new_node)

    def _del_right(self, col_index):
        ''' (MatrixNode, int) -> NoneType
            Travel right through the list and delete a node at col_index
        '''
        prev = None
        if(self.get_right()):
            while(self.get_right() is not None and
                  self.get_right().get_col() <= col_index):
                prev = self
                self = self.get_right()

                if(self.get_col() == col_index):
                    if(self.get_right()):
                        prev.set_right(self.get_right())
                    else:
                        prev.set_right(None)

    def _del_down(self, row_index):
        ''' (MatrixNode, int) -> NoneType
            Travel right through the list and delete a node at col_index
        '''
        prev = None
        if(self.get_down()):
            while(self.get_down() is not None and
                  self.get_down().get_row() <= row_index):
                prev = self
                self = self.get_down()

                if(self.get_row() == row_index):
                    if(self.get_down()):
                        prev.set_down(self.get_down())
                    else:
                        prev.set_down(None)

    def search_right(self, col_index):
        ''' (MatrixNode, int) -> MatrixNode or NoneType
            Return the value at the col index
        '''
        # if this node is the index
        node = None
        while(self.get_right() is not None and self.get_col() < col_index):
            self = self.get_right()
            if(self.get_col() == col_index):
                node = self
        return node

    def search_down(self, row_index):
        ''' (MatrixNode, int) -> MatrixNode or NoneType
            Return the value at the row index
        '''
        # if this node is the index
        node = None
        while(self.get_down() is not None and self.get_row() < row_index):
            self = self.get_down()
            if(self.get_row() == row_index):
                node = self
        return node

    def get_contents(self):
        '''(MatrixNode) -> obj
            Return the contents of this node
        '''
        return self._contents

    def set_contents(self, new_contents):
        '''(MatrixNode, obj) -> NoneType
            Set the contents of this node to new_contents
        '''
        self._contents = new_contents

    def get_right(self):
        '''(MatrixNode) -> MatrixNode
            Return the node to the right of this one
        '''
        return self._right

    def set_right(self, new_node):
        '''(MatrixNode, MatrixNode) -> NoneType
            Set the new_node to be to the right of this one in the matrix
        '''
        self._right = new_node

    def get_down(self):
        '''(MatrixNode) -> MatrixNode
            Return the node below this one
        '''
        return self._down

    def set_down(self, new_node):
        '''(MatrixNode, MatrixNode) -> NoneType
            Set new_node to be below this one in the matrix
        '''
        self._down = new_node


class Matrix():
    '''A class to represent a mathematical matrix'''

    def __init__(self, m, n, default=0):
        '''(Matrix, int, int, float) -> NoneType
            Create a new m x n matrix with all values set to default
        '''
        self._head = MatrixNode(None, row=-1, col=-1)
        self._default = default
        self._rows = m
        self._cols = n

    def search_index(self, row, col):
        ''' (Matrix, int, int) -> MatrixNode or NoneType
            Get contents of item at position (row, col)
        '''
        if(row < 0 or col < 0):
            raise MatrixIndexError("Index cannot be negative")
        elif(row > self.num_rows() or col > self.num_cols()):
            raise MatrixIndexError("Index out of bounds")

        col_node = self._head.search_right(col)
        if(isinstance(col_node, MatrixNode)):
            row_col_node = col_node.search_down(row)
            if(isinstance(row_col_node, MatrixNode)):
                return row_col_node

    def get_val(self, i, j):
        '''(Matrix, int, int) -> float
            Return the value of m[i,j] for this matrix m
        '''
        # if no value is found, we will return the default value
        val = self._default

        item = self.search_index(i, j)
        # if we found the item, overwrite the default value
        if(isinstance(item, MatrixNode)):
            val = item.get_contents()

        return val

    def _delete_index(self, i, j):
        ''' (Matrix, int, int) -> NoneType
            Deletes an item at (i, j)
        '''
        # if a column index node was requested to be deleted
        if(i == -1):
            self._head._del_right(j)
        # if a row index node was requested to be deleted
        elif(j == -1):
            self._head._del_down(i)
        else:
            # find the index nodes
            i_index_node = self._head.search_down(i)
            j_index_node = self._head.search_right(j)

            # delete the node at the intersection (i, j)
            i_index_node._del_right(j)
            j_index_node._del_down(i)

    def _del_col(self, j):
        '''(Matrix, int) -> NoneType
            Deletes column j
        '''
        j_node = self._head.search_right(j)
        current = j_node._down
        while (j_node.get_down()):
            self._delete_index(current.get_row(), j)
            current = j_node.get_down()
        # delete index node
        self._delete_index(-1, j)

    def _del_row(self, i):
        '''(Matrix, int) -> NoneType
            Deletes row i
        '''
        i_node = self._head.search_down(i)
        current = i_node.get_right()
        while (i_node.get_right()):
            self._delete_index(i, current.get_col())
            current = i_node.get_right()
        # delete index node
        self._delete_index(i, -1)

    def set_val(self, i, j, new_val):
        '''(Matrix, int, int, float) -> NoneType
            Set the value of m[i,j] to new_val for this matrix m
        '''
        if(i < 0 or j < 0):
            raise MatrixIndexError("Index cannot be negative")
        elif(i > self.num_rows() or j > self.num_cols()):
            raise MatrixIndexError("Index out of bounds")

        # check if the index nodes exist
        i_node = self._head.search_down(i)
        j_node = self._head.search_right(j)

        # if the row index node don't exist, create it
        if(not i_node):
            i_node = MatrixNode(i, row=i, col=-1)
            self._head._add_down(i_node, i)
        # if the column index node don't exist, create it
        if(not j_node):
            j_node = MatrixNode(j, row=-1, col=j)
            self._head._add_right(j_node, j)

        # find if item exists at index (i, j)
        item = self.search_index(i, j)

        if(isinstance(item, MatrixNode)):
            # delete the default value, it isn't neccessary to store
            if(new_val == self._default):
                self._delete_index(i, j)
            else:
                # since the node already exists, just overwrite the value
                item.set_contents(new_val)
        # if the node doesn't exist, create one
        else:
            new_node = MatrixNode(new_val, row=i, col=j)
            # attach the node to the index nodes
            i_node._add_right(new_node, j)
            j_node._add_down(new_node, i)

    def get_row(self, row_num):
        '''(Matrix, int) -> OneDimensionalMatrix
            Return the row_num'th row of this matrix
        '''
        if(row_num not in range(self.num_rows())):
            raise MatrixIndexError("row selected is out of bounds")

        new_col = OneDimensionalMatrix(self.num_cols(), self._default)
        curr_index = self._head.search_down(row_num)

        if(isinstance(curr_index, MatrixNode)):
            curr_index = curr_index.get_right()
            while(curr_index):
                new_col.set_item(
                    curr_index.get_col(),
                    curr_index.get_contents())
                curr_index = curr_index.get_right()
        return new_col

    def set_row(self, row_num, new_row):
        '''(Matrix, int, OneDimensionalMatrix) -> NoneType
            Set the value of the row_num'th row of this matrix to
            those of new_row
        '''
        if(row_num not in range(self.num_rows())):
            raise MatrixIndexError("row selected is out of bounds")
        elif(self.num_cols() != new_row.num_cols()):
            raise MatrixDimensionError("row doesn't fit into the matrix")

        curr_index = self._head.search_down(row_num)
        row_index = new_row._head.get_down()

        # check if the row node exists in this matrix
        if(isinstance(curr_index, MatrixNode)):
            if(row_index is None):
                self._del_row(row_num)
            elif(row_index is not None):
                curr_index = curr_index.get_right()
                row_index = row_index.get_right()
                while(curr_index and row_index):
                    # update the current matrix element
                    if(curr_index.get_col() == row_index.get_col()):
                        curr_index.set_contents(row_index.get_contents())
                        curr_index = curr_index.get_right()
                        row_index = row_index.get_right()
                    # delete elements if they're not part of the new row
                    elif(curr_index.get_col() < row_index.get_col()):
                        self._delete_index(row_num, curr_index.get_col())
                        curr_index = curr_index.get_right()
                    else:
                        self.set_val(
                            row_num,
                            row_index.get_col(),
                            row_index.get_contents())
                        row_index = row_index.get_right()
                # if the row matrix has no more nodes left, delete elements
                # in the matrix that the row matrix doesn't have
                while (curr_index and not (row_index)):
                    self._delete_index(row_num, curr_index.get_col())
                    curr_index = curr_index.get_right()
                # if this matrix has no nodes left, set all the rows elements
                # to this matrix
                while (not (curr_index) and row_index):
                    self.set_val(
                        row_num,
                        row_index.get_col(),
                        row_index.get_contents())
                    row_index = row_index.get_right()
        # this row node does not exist
        else:
            if(row_index is not None):
                row_index = row_index.get_right()
                # put the every element of the row to the matrix
                while(row_index):
                    self.set_val(
                        row_num,
                        row_index.get_col(),
                        row_index.get_contents())
                    row_index = row_index.get_right()

    def get_col(self, col_num):
        '''(Matrix, int) -> OneDimensionalMatrix
            Return the col_num'th column of this matrix
        '''
        if(col_num not in range(self.num_cols())):
            raise MatrixIndexError("column selected is out of bounds")

        new_col = OneDimensionalMatrix(self.num_rows(), self._default)
        curr_index = self._head.search_right(col_num)

        if(isinstance(curr_index, MatrixNode)):
            curr_index = curr_index.get_down()
            while(curr_index):
                new_col.set_item(
                    curr_index.get_row(),
                    curr_index.get_contents())
                curr_index = curr_index.get_down()
        return new_col

    def set_col(self, col_num, new_col):
        '''(Matrix, int, OneDimensionalMatrix) -> NoneType
            Set the value of the col_num'th column of this matrix to those
            of new_row
        '''
        if(col_num not in range(self.num_cols())):
            raise MatrixIndexError("column selected is out of bounds")
        elif(self.num_rows() != new_col.num_cols()):
            raise MatrixDimensionError("column doesn't fit into the matrix")

        curr_index = self._head.search_right(col_num)
        col_index = new_col._head.get_down()

        # check if the column node exists in this matrix
        if(isinstance(curr_index, MatrixNode)):
            # if the column is empty
            if(col_index is None):
                self._del_col(col_num)
            elif(col_index is not None):
                curr_index = curr_index.get_down()
                col_index = col_index.get_right()
                while(curr_index and col_index):
                    # update the current matrix element
                    if(curr_index.get_row() == col_index.get_col()):
                        curr_index.set_contents(col_index.get_contents())
                        curr_index = curr_index.get_down()
                        col_index = col_index.get_right()
                    # delete elements if they're not part of the new column
                    elif(curr_index.get_row() < col_index.get_col()):
                        self._delete_index(curr_index.get_row(), col_num)
                        curr_index = curr_index.get_down()
                    else:
                        self.set_val(
                            col_index.get_col(),
                            col_num,
                            col_index.get_contents())
                        col_index = col_index.get_right()
                # if the column matrix has no more nodes left, delete elements
                # in the matrix that the column matrix doesn't have
                while (curr_index and not (col_index)):
                    self._delete_index(curr_index.get_row(), col_num)
                    curr_index = curr_index.get_down()
                # if this matrix has no nodes left, set all the columns
                # elements to this matrix
                while (not (curr_index) and col_index):
                    self.set_val(
                        col_index.get_col(),
                        col_num,
                        col_index.get_contents())
                    col_index = col_index.get_right()
        # this column node does not exist
        else:
            if(col_index is not None):
                col_index = col_index.get_right()
                # put the every element of the column to the matrix
                while(col_index):
                    self.set_val(
                        col_index.get_col(),
                        col_num,
                        col_index.get_contents())
                    col_index = col_index.get_right()

    def swap_rows(self, i, j):
        '''(Matrix, int, int) -> NoneType
            Swap the values of rows i and j in this matrix
        '''
        if (i not in range(self.num_rows())):
            raise MatrixIndexError('row i is out of bounds')
        elif(j not in range(self.num_rows())):
            raise MatrixIndexError('row j is out of bounds')

        i_row = self.get_row(i)
        j_row = self.get_row(j)

        self.set_row(i, j_row)
        self.set_row(j, i_row)

    def swap_cols(self, i, j):
        '''(Matrix, int, int) -> NoneType
            Swap the values of columns i and j in this matrix
        '''
        if (i not in range(self.num_cols())):
            raise MatrixIndexError('column i is out of bounds')
        elif(j not in range(self.num_cols())):
            raise MatrixIndexError('column j is out of bounds')

        i_col = self.get_col(i)
        j_col = self.get_col(j)

        self.set_col(i, j_col)
        self.set_col(j, i_col)

    def add_scalar(self, add_value):
        '''(Matrix, float) -> NoneType
            Increase all values in this matrix by add_value
        '''
        # update the default value
        self._default += add_value

        i_node = self._head.get_down()
        while(i_node):
            node = i_node.get_right()
            while(node):
                new_val = node.get_contents() + add_value
                node.set_contents(new_val)
                node = node.get_right()
            i_node = i_node.get_down()

    def subtract_scalar(self, sub_value):
        '''(Matrix, float) -> NoneType
            Decrease all values in this matrix by sub_value
        '''
        # formula for difference between two matrices
        self.add_scalar((-1) * sub_value)

    def multiply_scalar(self, mult_value):
        '''(Matrix, float) -> NoneType
            Multiply all values in this matrix by mult_value
        '''
        # update the default value
        self._default *= mult_value

        i_node = self._head.get_down()
        while(i_node):
            node = i_node.get_right()
            while(node):
                new_val = node.get_contents() * mult_value
                node.set_contents(new_val)
                node = node.get_right()
            i_node = i_node.get_down()

    def print_matrix(self):
        ''' (Matrix) -> NoneType
            A simple representation of the matrix
        '''
        for i in range((self.num_rows())):
            print(" ")
            for j in range((self.num_cols())):
                print(self.get_val(i, j), " ", end="")

    def add_matrix(self, adder_matrix):
        '''(Matrix, Matrix) -> Matrix
            Return a new matrix that is the sum of this matrix and adder_matrix 
        '''
        if(self.num_rows() != adder_matrix.num_rows() or
           self.num_cols() != adder_matrix.num_cols()):
            raise MatrixDimensionError("cannot add matrices because matrix1 " +
                                       "dimensions != matrix2 dimensions")
        # update the default values
        new_default = self._default + adder_matrix._default

        new_matrix = Matrix(self.num_rows(), self.num_cols(), new_default)

        # keep track of the row index on both matrices
        index_a = self._head.get_down()
        index_b = adder_matrix._head.get_down()

        while(index_a and index_b):
            # check if we're on the same row in both matrices
            if(index_a.get_row() == index_b.get_row()):
                # imaginary pointers for every column in the row
                right_a = index_a.get_right()
                right_b = index_b.get_right()
                while(right_a and right_b):
                    # if the positions are the same
                    if(right_a.get_col() == right_b.get_col()):
                        new_val = (right_a.get_contents() + 
                                   right_b.get_contents())
                        new_matrix.set_val(
                            right_a.get_row(), right_a.get_col(), new_val)
                        right_a = right_a.get_right()
                        right_b = right_b.get_right()
                    elif(right_a.get_col() < right_b.get_col()):
                        # right_a try to catch up with right_b while appending
                        # its own data
                        new_matrix.set_val(
                            right_a.get_row(),
                            right_a.get_col(),
                            right_a.get_contents())
                        right_a = right_a.get_right()
                    else:
                        # right_b try to catch up with right_a while appending
                        # its own` data
                        new_matrix.set_val(
                            right_b.get_row(),
                            right_b.get_col(),
                            right_b.get_contents())
                        right_b = right_b.get_right()
                # if row a exists but has no nodes, append b nodes
                while(not(right_a) and right_b):
                    new_matrix.set_val(
                        right_b.get_row(),
                        right_b.get_col(),
                        right_b.get_contents())
                    right_b = right_b.get_right()
                # if row b exists but has no nodes, append a nodes
                while(right_a and not(right_b)):
                    new_matrix.set_val(
                        right_a.get_row(),
                        right_a.get_col(),
                        right_a.get_contents())
                    right_a = right_a.get_right()
                # move row down on both matrices
                index_a = index_a.get_down()
                index_b = index_b.get_down()
            # append the rows of index_a until caught up with index_b
            elif(index_a.get_row() < index_b.get_row()):
                right_a = index_a.get_right()
                while(right_a):
                    new_matrix.set_val(
                        right_a.get_row(),
                        right_a.get_col(),
                        right_a.get_contents())
                    right_a = right_a.get_right()
                index_a = index_a.get_down()
            else:
                right_b = index_b.get_right()
                while(right_b):
                    new_matrix.set_val(
                        right_b.get_row(),
                        right_b.get_col(),
                        right_b.get_contents())
                    right_b = right_b.get_right()
                index_b = index_b.get_down()
        # if row a doesn't exist append row b to the new matrix
        while(not(index_a) and index_b):
            right_b = index_b.get_right()
            while(right_b):
                new_matrix.set_val(
                    right_b.get_row(),
                    right_b.get_col(),
                    right_b.get_contents())
                right_b = right_b.get_right()
            index_b = index_b.get_down()
        # if row b doesn't exist append row a to the new matrix
        while(index_a and not(index_b)):
            right_a = index_a.get_right()
            while(right_a):
                new_matrix.set_val(
                    right_a.get_row(),
                    right_a.get_col(),
                    right_a.get_contents())
                right_a = right_a.get_right()
            index_a = index_a.get_down()
        return new_matrix

    def multiply_matrix(self, mult_matrix):
        '''(Matrix, Matrix) -> Matrix
            Return a new matrix that is the product of this matrix and mult_matrix
        '''
        if (self.num_cols() != mult_matrix.num_rows()):
            raise MatrixDimensionError(
                "matrix1 column size != matrix2 row size")

        new_matrix = Matrix(self.num_cols(), mult_matrix.num_rows())

        for i in range(self.num_rows()):
            # for every row of matrix1, go through every column of matrix2
            for col in range(mult_matrix.num_cols()):
                # reset the value
                new_val = 0
                for j in range(self.num_cols()):
                    new_val += (self.get_val(i, j) *
                                mult_matrix.get_val(j, col))
                # set new multiplied value
                new_matrix.set_val(i, col, new_val)
        return new_matrix

    def num_rows(self):
        return self._rows

    def num_cols(self):
        return self._cols


class OneDimensionalMatrix(Matrix):
    ''' A 1xn or nx1 matrix. '''

    def __init__(self, n, default=0):
        Matrix.__init__(self, 1, n, default)

    def get_item(self, i):
        '''(OneDimensionalMatrix, int) -> float
            Return the i'th item in this matrix
        '''
        return self.get_val(0, i)

    def set_item(self, i, new_val):
        '''(OneDimensionalMatrix, int, float) -> NoneType
            Set the i'th item in this matrix to new_val
        '''
        self.set_val(0, i, new_val)


class SquareMatrix(Matrix):
    ''' A matrix where the number of rows and columns are equal '''

    def __init__(self, n):
        '''(SquareMatrix, int) -> NoneType
            Create an nxn Square Matrix with all the values set to default
        '''
        Matrix.__init__(self, n, n, default=0)

    def transpose(self):
        '''(SquareMatrix) -> NoneType
            Transpose this matrix
        '''
        temp_matrix = SquareMatrix(self.num_rows())
        # transpose the matrix
        for n1 in range(self.num_rows()):
            for n2 in range(self.num_rows()):
                temp_matrix.set_val(n2, n1, self.get_val(n1, n2))
        # apply the transpose to the current matrix
        for n1 in range(self.num_rows()):
            for n2 in range(self.num_rows()):
                self.set_val(n1, n2, temp_matrix.get_val(n1, n2))

    def get_diagonal(self):
        '''(Squarematrix) -> OneDimensionalMatrix
            Return a one dimensional matrix with the values of the diagonal
            of this matrix
        '''
        m1d = OneDimensionalMatrix(self.num_rows())
        for n in range(self.num_rows()):
            m1d.set_item(n, self.get_val(n, n))
        return m1d

    def set_diagonal(self, new_diagonal):
        '''(SquareMatrix, OneDimensionalMatrix) -> NoneType
        Set the values of the diagonal of this matrix to those of new_diagonal
        '''
        for n in range(new_diagonal.num_cols()):
            self.set_val(n, n, new_diagonal.get_item(n))


class SymmetricMatrix(SquareMatrix):
    ''' A Symmetric Matrix, where m[i, j] = m[j, i] for all i and j '''

    def set_val(self, i, j, new_val):
        '''(SymmetricMatrix, int, int, int) -> NoneType
            Preserve the symmetry when setting value
        '''
        Matrix.set_val(self, i, j, new_val)
        Matrix.set_val(self, j, i, new_val)

    def set_row(self, n, new_row):
        '''(SymmetricMatrix, int, OneDimensionalMatrix) -> NoneType
            Preserve symmetry when setting row
        '''
        Matrix.set_row(self, n, new_row)
        Matrix.set_col(self, n, new_row)

    def set_col(self, m, new_col):
        '''(SymmetricMatrix, int, OneDimensionalMatrix) -> NoneType
            Preserve symmetry when setting column
        '''
        Matrix.set_row(self, n, new_row)
        Matrix.set_col(self, n, new_row)

    def transpose(self):
        '''(SymmetricMatrix) -> NoneType
            Transposes the matrix
        '''
        # The matrix would end up being the same if we transpose it
        # so avoid extra calculations
        pass


class DiagonalMatrix(SquareMatrix, OneDimensionalMatrix):
    ''' A square matrix with 0 values everywhere but the diagonal '''

    def __init__(self, n):
        '''(DiagonalMatrix, int) -> NoneType
            Creates an nxn diagonal matrix
        '''
        SquareMatrix.__init__(self, n)

    def get_val(self, n):
        '''(DiagonalMatrix, int) -> float
            Gets the (n,n) item of the diagonal
        '''
        return self.get_val(n, n)

    def set_val(self, n, new_val):
        '''(DiagonalMatrix, int, int) -> NoneType
            Sets the (n,n) item on the diagonal
        '''
        self.set_val(n, n, new_val)

    def set_item(self, n, new_val):
        '''(DiagonalMatrix, int, int) -> NoneType
            Sets the (n, n) item of the diagonal
        '''
        Matrix.set_val(self, n, n, new_val)

    def transpose(self):
        '''(SymmetricMatrix) -> NoneType
            Transposes this matrix
        '''
        # Nothing will happen to the diagonal if transposed
        pass


class IdentityMatrix(DiagonalMatrix):
    '''A matrix with 1s on the diagonal and 0s everywhere else'''

    def __init__(self, n):
        '''(IdentityMatrix, int) -> NoneType
            Creates an nxn Identity Matrix
        '''
        DiagonalMatrix.__init__(self, n)

        for i in range(n):
            self.set_item(i, 1)

    def add_scalar(self, add_value):
        '''(IdentityMatrix, float) -> NoneType
            Invalid operation.
        '''
        raise MatrixInvalidOperationError(
            'Cannot add scalar to identity matrix')

    def multiply_scalar(self, mult_value):
        '''(IdentityMatrix, float) -> NoneType
            Invalid operation.
        '''
        raise MatrixInvalidOperationError(
            'Cannot multiply identity matrix by a scalar')
